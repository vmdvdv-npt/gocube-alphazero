from concurrent.futures import ThreadPoolExecutor
import json
import threading

from gocube_golden import process_supervision as supervision


def test_same_process_concurrent_heartbeat_publication(tmp_path, monkeypatch):
    path = tmp_path / "heartbeat.json"
    barrier = threading.Barrier(2)
    real_replace = supervision.os.replace
    def simultaneous_replace(source, destination):
        barrier.wait(timeout=5)
        real_replace(source, destination)
    monkeypatch.setattr(supervision.os, "replace", simultaneous_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(supervision.atomic_write_json, path, {"progress": n}) for n in (1, 2)]
        for future in futures:
            future.result(timeout=10)
    assert json.loads(path.read_text()) in ({"progress": 1}, {"progress": 2})
    assert not list(tmp_path.glob("*.tmp"))
