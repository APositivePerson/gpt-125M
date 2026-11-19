"""TLS-verified curl transport and offline readers for public SFT data."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import quote, urlsplit
from progress_status import stage

ROOT = Path(__file__).resolve().parent
REPO = 'HuggingFaceTB/smol-smoltalk'


def endpoint_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('--endpoint must be an HTTPS base URL without credentials, query or fragment')
    return value.rstrip('/')


def curl_base():
    binary = shutil.which('curl')
    if not binary:
        raise RuntimeError('找不到 curl；可安装 curl，或使用 --backend hub / --local-data。')
    command = [binary, '--fail', '--location', '--retry', '3', '--retry-delay', '2',
               '--retry-all-errors', '--connect-timeout', '10', '--proto', '=https', '--proto-redir', '=https']
    if os.environ.get('SFT_CURL_PROXY'):
        command.extend(['--proxy', os.environ['SFT_CURL_PROXY']])
    return command


def fetch_info(endpoint, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'repo_info.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        print(f'复用已固定的数据版本：{info["sha"]}', flush=True)
    else:
        with stage(f'通过 curl 查询数据版本：{endpoint}（保留 SSL 校验，失败重试 3 次）'):
            result = subprocess.run([*curl_base(), '--silent', '--show-error', '--max-time', '120',
                                     f'{endpoint}/api/datasets/{REPO}'], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f'curl 无法连接数据源（退出码 {result.returncode}）：{result.stderr.strip()[:600]}\n'
                                   '检查网络或代理；也可通过 --endpoint 指定可访问的 HTTPS 数据源，或用 --local-data 读取本地数据。')
            info = json.loads(result.stdout)
    if not re.fullmatch(r'[0-9a-f]{40,64}', info.get('sha', '')) or not isinstance(info.get('siblings'), list):
        raise RuntimeError('数据接口返回的版本或文件列表无效')
    if not manifest.exists():
        tmp = manifest.with_suffix('.tmp')
        tmp.write_text(json.dumps(info))
        os.replace(tmp, manifest)
    return info


def training_files(info):
    files = sorted(s['rfilename'] for s in info['siblings'] if s.get('rfilename', '').endswith('.parquet')
                   and 'train' in Path(s['rfilename']).name.lower())
    if not files:
        raise RuntimeError('没有找到训练集 Parquet 文件；请检查数据仓库文件列表。')
    for name in files:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('数据源包含非法文件路径')
    return files


def parquet_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['messages']):
        yield from batch.to_pylist()


def download_shard(endpoint, revision, name, part, attempts=3):
    """Resolve a fresh signed URL on every attempt; never log signed redirects."""
    base = f'{endpoint}/datasets/{REPO}/resolve/{revision}/{quote(name, safe="/")}'
    last_status = '000'
    for attempt in range(1, attempts + 1):
        # Keep the data revision fixed while refreshing cached signed redirects.
        url = f'{base}?download=true&cache_bust={time.time_ns()}'
        result = subprocess.run([*curl_base(), '--retry', '0', '--progress-bar', '--max-time', '1800',
                                 '--header', 'Cache-Control: no-cache', '--continue-at', '-',
                                 '--output', str(part), '--write-out', '%{http_code}', url],
                                stdout=subprocess.PIPE, text=True)
        last_status = (result.stdout or '000').strip()[-3:]
        if result.returncode == 0 and last_status in ('200', '206'):
            return
        if last_status == '416' and part.exists():
            import pyarrow.parquet as pq
            try:
                pq.ParquetFile(part)
                return
            except Exception:
                pass
        if result.returncode == 33 or last_status == '416':
            if part.exists():
                preserved = part.with_name(part.name + f'.restart-{time.time_ns()}')
                os.replace(part, preserved)
                print(f'服务器拒绝续传，旧片段保留为 {preserved.name}；将重新下载。', flush=True)
        if last_status in ('401', '403'):
            print(f'文件存储地址返回 HTTP {last_status}；重新获取下载链接（{attempt}/{attempts}）。', flush=True)
        else:
            print(f'分片请求失败：HTTP {last_status}，curl={result.returncode}（{attempt}/{attempts}）。', flush=True)
        if last_status == '404':
            break
        if attempt < attempts:
            time.sleep(2)
    raise RuntimeError(f'分片仍无法下载：HTTP {last_status}。已保留下载片段。\n'
                       '主页可访问不代表文件 CDN 可访问；请检查代理对 huggingface.co 及其重定向存储域名的连通性。\n'
                       '也可在其他可联网环境下载训练 Parquet，再通过 --local-data 读取。')


def curl_rows(endpoint, cache, info):
    import pyarrow.parquet as pq
    for index, name in enumerate(training_files(info), 1):
        path = Path(cache) / info['sha'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            with stage(f'下载第 {index} 个训练分片：{name}（支持续传）'):
                download_shard(endpoint, info['sha'], name, part)
                pq.ParquetFile(part)  # Verify it is a readable Parquet before publishing.
                os.replace(part, path)
        print(f'读取本地分片：{path.name}', flush=True)
        yield from parquet_rows(path)


def local_files(source):
    source = Path(source).resolve()
    files = sorted(source.glob('*.parquet')) if source.is_dir() else [source]
    if not files or any(not p.is_file() or p.suffix not in ('.parquet', '.jsonl') for p in files):
        raise ValueError('--local-data must be a .jsonl/.parquet file or a directory containing training .parquet files')
    return files


def local_identity(files):
    identity = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def local_rows(files):
    for path in files:
        print(f'读取本地数据：{path}', flush=True)
        if path.suffix == '.parquet':
            yield from parquet_rows(path)
        else:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
