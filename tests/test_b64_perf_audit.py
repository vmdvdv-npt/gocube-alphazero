import json
import pytest

from gocube_golden.training_profile import Collector, measured, span
from gocube_golden.torus9_five_channel_training import TrainingHeartbeat
from gocube_golden.b64_perf_audit import run, stats


def test_collector_scope_resets_after_exception():
    collector = Collector()
    with pytest.raises(RuntimeError), collector.activate(), span('outer'):
        with span('inner'):
            raise RuntimeError('stop')
    assert {'inner', 'outer'} == collector.finish().keys()
    assert measured('off', lambda: 42) == 42
    assert len(collector.rows) == 2


def test_real_heartbeat_retains_two_fsyncs_and_serialization(tmp_path):
    from unittest.mock import patch
    import os
    from gocube_golden import process_supervision
    collector = Collector()
    heartbeat = TrainingHeartbeat(tmp_path/'heartbeat.json', 256)
    real_fsync = os.fsync
    with patch.object(process_supervision.os, 'fsync', wraps=real_fsync) as fsync, collector.activate():
        heartbeat.mark('training', 1, 2560)
    assert fsync.call_count == 2
    payload = json.loads(heartbeat.path.read_text())
    assert payload['progress_token'] == 'training:1'
    assert payload['generation'] == 256
    assert payload['total'] == 2560
    assert {'mark','heartbeat','io.json','io.file_fsync','io.directory_fsync'} <= collector.finish().keys()
    assert not list(tmp_path.glob('*.tmp'))


def test_audit_rejects_output_inside_production_without_writes(tmp_path):
    runs = tmp_path/'runs'; runs.mkdir()
    with pytest.raises(ValueError, match='outside'):
        run({'schema':'gocube-b64-perf-audit-v1', 'runs_root':str(runs), 'output':str(runs/'new')})
    assert list(runs.iterdir()) == []


def test_statistics_interpolate_percentiles():
    result = stats([1,2,3,4])
    assert result['p50'] == 2.5
    assert result['p95'] == pytest.approx(3.85)
    assert stats([]) is None


def test_nsight_attribution_uses_launch_correlation_and_counts_only_actual_waits(tmp_path):
    import sqlite3
    from gocube_golden.b64_perf_report import nsight_summary
    path=tmp_path/'trace.sqlite'
    with sqlite3.connect(path) as c:
        c.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,shortName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER);
            INSERT INTO StringIds VALUES (1,'cudaLaunchKernel'),(2,'cudaStreamSynchronize'),(3,'cudaMemcpyAsync'),(4,'tiny kernel');
            INSERT INTO NVTX_EVENTS VALUES (0,1000,7,'adam',NULL),(1000,2000,7,'backward',NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10,20,7,11,1),(30,50,7,12,3),(60,90,7,13,2),(1100,1150,8,14,1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100,200,11,4),(3000,3100,14,4);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (210,230,12);
        ''')
    result=nsight_summary(path, {'updates':1,'loop_wall_ms':1.0})
    assert result['kernel_count']==2
    assert result['spans']['backward']['kernels_per_update']==1
    assert result['spans']['adam']['kernels_per_update']==1
    assert result['spans']['adam']['sync_calls_per_update']==1
    assert result['spans']['adam']['sync_cpu_ms_per_update']==pytest.approx(30/1e6)
    assert result['spans']['adam']['transfer_ms_per_update']==pytest.approx(20/1e6)
