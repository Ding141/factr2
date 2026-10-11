"""Exercise slow/failing audit storage without ROS or robot commands."""
import json
import threading
import time

import pytest

from w3_motion_audit import MotionAuditWriter


def test_blocked_disk_does_not_block_motion_and_preserves_last_complete_snapshot(tmp_path):
    entered = threading.Event(); release = threading.Event(); writes = []
    def slow_write(path, report):
        entered.set()
        assert release.wait(2.)
        writes.append(report)
        path.write_text(json.dumps(report))
    writer = MotionAuditWriter(tmp_path/'audit.json', write=slow_write, log=lambda *a, **k:None)
    report = dict(status='running',events=[dict(index=1,name='sweep',status='sending')])
    try:
        writer.submit(report)
        assert entered.wait(1.)
        report['events'][0]['status']='completed'
        started=time.monotonic()
        writer.submit(report)
        assert time.monotonic()-started < .1
        assert writer.pending['events'][0]['status']=='completed'
        report['status']='completed';writer.submit(report)
        report['status']='caller_mutated_after_submit'
        release.set();writer.flush()
        assert writes[0]['events'][0]['status']=='sending'
        saved=json.loads((tmp_path/'audit.json').read_text())
        assert saved['status']=='completed' and saved['events'][0]['status']=='completed'
    finally:
        release.set();writer.close()
    assert not writer.thread.is_alive()


def test_write_failure_propagates_to_motion_guard(tmp_path):
    def bad_write(*args):raise OSError('disk full')
    writer=MotionAuditWriter(tmp_path/'audit.json',write=bad_write)
    writer.submit(dict(events=[]))
    with pytest.raises(RuntimeError,match='disk full'):writer.flush()
    with pytest.raises(RuntimeError,match='disk full'):writer.check()
    with pytest.raises(RuntimeError,match='disk full'):writer.close()
    assert not writer.thread.is_alive()


def test_stalled_writer_aborts_instead_of_accumulating_unbounded_audits(tmp_path):
    entered=threading.Event();release=threading.Event()
    def blocked(*args):entered.set();release.wait(2.)
    writer=MotionAuditWriter(tmp_path/'audit.json',write=blocked,max_write_seconds=.05)
    try:
        writer.submit(dict(events=[]));assert entered.wait(1.)
        time.sleep(.07)
        with pytest.raises(RuntimeError,match='stalled'):writer.check()
    finally:
        release.set();writer.thread.join(.1);writer.close()
