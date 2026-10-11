"""Bounded asynchronous audit output; never perform disk/stdout I/O in motion ticks."""
import copy
import threading
import time

from w3_motion_sequence import write_json


class MotionAuditWriter:
    def __init__(self, path, write=write_json, log=print, max_write_seconds=5.):
        self.path = path
        self.write = write
        self.log = log
        self.max_write_seconds = max_write_seconds
        self.condition = threading.Condition()
        self.pending = None
        self.writing_since = None
        self.error = None
        self.stopping = False
        self.logged_events = 0
        self.thread = threading.Thread(target=self._run, name='motion_audit_writer', daemon=True)
        self.thread.start()

    def _check_locked(self):
        if self.error is not None:
            raise RuntimeError('Motion audit writer failed: '+str(self.error)) from self.error
        if self.writing_since is not None and time.monotonic()-self.writing_since > self.max_write_seconds:
            raise RuntimeError('Motion audit writer stalled >5s; stop motion')

    def check(self):
        with self.condition:
            self._check_locked()

    def submit(self, report):
        # One producer owns report. Snapshot before returning; worker never reads
        # mutable controller state. Replacing a pending snapshot preserves all
        # accumulated events while bounding memory even if storage is slow.
        snapshot = copy.deepcopy(report)
        with self.condition:
            self._check_locked()
            if self.stopping:
                raise RuntimeError('Motion audit writer already closed')
            self.pending = snapshot
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopping)
                if self.pending is None:
                    return
                snapshot = self.pending
                self.pending = None
                self.writing_since = time.monotonic()
            try:
                self.write(self.path, snapshot)
                events = snapshot.get('events', [])
                for event in events[self.logged_events:]:
                    self.log(f'[{event["index"]}] {event["name"]}', flush=True)
                self.logged_events = len(events)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.pending = None
                    self.stopping = True
                    self.writing_since = None
                    self.condition.notify_all()
                return
            with self.condition:
                self.writing_since = None
                self.condition.notify_all()

    def flush(self, timeout=10.):
        deadline = time.monotonic()+timeout
        with self.condition:
            while self.pending is not None or self.writing_since is not None:
                self._check_locked()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Motion audit final flush timed out')
                self.condition.wait(min(.1, remaining))
            self._check_locked()

    def close(self):
        try:
            self.flush()
        finally:
            with self.condition:
                self.stopping = True
                self.condition.notify_all()
            self.thread.join(timeout=1.)
