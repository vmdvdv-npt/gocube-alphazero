"""Isolated monitor accounting and GPU parse checks; no training or transport."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('monitor', Path(__file__).parents[1] / 'tools/training_efficiency_monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def test_missing_final_update_does_not_invent_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'replay').mkdir()
            rolling = root / 'replay/rolling-after-254.jsonl'
            rolling.write_text('[]')
            (root / 'training').mkdir()
            (root / 'training/iter-254.json').write_text(json.dumps({'updates': [{'batch_size': 128}] * 10}))
            start = rolling.stat().st_mtime
            samples = [{'sample_started_at': start, 'heartbeat': {'phase': 'training', 'done': 9, 'progress_at': start + 5}, 'gpu': {'status': 'unavailable'}}]
            result = monitor.summarize(samples, root, 254, {})
            self.assertIsNone(result['updates_per_minute'])
            self.assertEqual(result['training_examples_processed'], 1280)
            samples[0]['heartbeat']['done'] = 10
            result = monitor.summarize(samples, root, 254, {})
            self.assertEqual(result['updates_per_minute'], 120)
            self.assertEqual(result['examples_per_minute'], 15360)

    def test_unavailable_gpu_and_malformed_output_are_visible(self):
        self.assertEqual(monitor.gpu_sample(None)['status'], 'unavailable')
        with patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='bad', stderr='')):
            self.assertEqual(monitor.gpu_sample('fake')['status'], 'error')

    def test_gpu_na_is_not_zero(self):
        with patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='0, UUID, GPU, 40, N/A, 1000, 6000, 20, 60, 500, 810, P8\n', stderr='')):
            row = monitor.gpu_sample('fake')['gpus'][0]
            self.assertIsNone(row['utilization.memory'])
            self.assertEqual(row['utilization.gpu'], 40)


if __name__ == '__main__':
    unittest.main()
