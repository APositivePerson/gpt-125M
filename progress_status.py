"""Visible status for startup/I/O stages that have no measurable percentage."""
from contextlib import contextmanager
import threading
import time
from tqdm.auto import tqdm


@contextmanager
def stage(label, detail=None, interval=10):
    started = time.monotonic()
    done = threading.Event()
    tqdm.write(f'▶ {label}')
    def heartbeat():
        while not done.wait(interval):
            extra = f' · {detail()}' if detail else ''
            tqdm.write(f'… {label}，已用 {time.monotonic() - started:.0f} 秒{extra}')
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        tqdm.write(f'✗ {label}未完成（{time.monotonic() - started:.1f} 秒）')
        raise
    else:
        tqdm.write(f'✓ {label}完成（{time.monotonic() - started:.1f} 秒）')
    finally:
        done.set()
        thread.join()
