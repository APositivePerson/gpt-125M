"""Offline regression checks for TLS transport, local data and failure handling."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import launch_sft
from sft_download import endpoint_url, fetch_info, training_files, download_shard


class SFTDownloadTests(unittest.TestCase):
    def test_401_refreshes_resolve_url_instead_of_retrying_same_signature(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sft_download.subprocess.run') as run, \
             patch('sft_download.time.sleep'), patch('sft_download.time.time_ns', side_effect=[101, 102]):
            run.side_effect = [SimpleNamespace(returncode=22, stdout='401'), SimpleNamespace(returncode=0, stdout='206')]
            with redirect_stdout(io.StringIO()):
                download_shard('https://huggingface.co', 'a' * 40, 'data/train-00000.parquet', Path(tmp) / 'file.part')
            self.assertEqual(run.call_count, 2)
            urls = [call.args[0][-1] for call in run.call_args_list]
            self.assertTrue(all(url.startswith('https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/' + 'a' * 40 + '/') for url in urls))
            self.assertNotEqual(urls[0], urls[1])
            self.assertTrue(all('download=true&cache_bust=' in url for url in urls))
            self.assertTrue(all('--insecure' not in call.args[0] for call in run.call_args_list))

    def test_persistent_401_stops_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sft_download.subprocess.run') as run, patch('sft_download.time.sleep'):
            run.return_value = SimpleNamespace(returncode=22, stdout='401')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
                download_shard('https://huggingface.co', 'a' * 40, 'data/train.parquet', Path(tmp) / 'file.part')
            self.assertEqual(run.call_count, 3)

    def test_curl_metadata_pins_revision_and_keeps_tls_checks(self):
        info = dict(sha='a' * 40, siblings=[dict(rfilename='data/train-00000.parquet')])
        with tempfile.TemporaryDirectory() as tmp, patch('sft_download.subprocess.run') as run:
            run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(info), stderr='')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(fetch_info('https://huggingface.co', tmp), info)
                self.assertEqual(fetch_info('https://huggingface.co', tmp), info)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertNotIn('--insecure', command)
            self.assertNotIn('-k', command)
            self.assertIn('--retry-all-errors', command)
            self.assertEqual(command[-1], 'https://huggingface.co/api/datasets/HuggingFaceTB/smol-smoltalk')

    def test_curl_failure_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch('sft_download.subprocess.run') as run:
            run.return_value = SimpleNamespace(returncode=35, stdout='', stderr='SSL connect error')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'local-data'):
                fetch_info('https://huggingface.co', tmp)
            self.assertFalse((Path(tmp) / 'repo_info.json').exists())

    def test_endpoint_and_cache_paths_are_checked(self):
        self.assertEqual(endpoint_url('https://huggingface.co/'), 'https://huggingface.co')
        for url in ['http://huggingface.co', 'https://name:secret@example.com', 'https://example.com?x=y']:
            with self.assertRaises(ValueError):
                endpoint_url(url)
        with self.assertRaises(RuntimeError):
            training_files(dict(siblings=[dict(rfilename='../train-00000.parquet')]))

    def test_local_preparation_requires_no_dataset_download(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'train.jsonl'
            rows = [dict(messages=[dict(role='user', content=f'Question {i}: Say hello.'),
                                   dict(role='assistant', content=f'Hello! This is answer {i}.')]) for i in range(200)]
            source.write_text('\n'.join(json.dumps(r) for r in rows))
            out = Path(tmp) / 'prepared'
            result = subprocess.run([sys.executable, str(root / 'prepare_sft.py'), '--local-data', str(source),
                                     '--train-examples', '3', '--val-examples', '2', '--output', str(out)],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((out / 'train.npz').exists())
            self.assertTrue((out / 'val.npz').exists())
            self.assertEqual(json.loads((out / 'source.json').read_text())['dataset'], 'local conversations')

    def test_prepare_failure_never_launches_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'base.pt'
            base.write_bytes(b'test')
            with patch.object(sys, 'argv', ['launch_sft.py', '--base', str(base)]), \
                 patch('launch_sft.subprocess.run', return_value=SimpleNamespace(returncode=1)), \
                 patch('launch_sft.os.execv') as launch, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    launch_sft.main()
                self.assertEqual(stop.exception.code, 1)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
