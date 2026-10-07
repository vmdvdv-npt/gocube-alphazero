"""Opt-in measurement spans. No torch/CUDA import or event work when disabled.

CPU spans are inclusive; consumers must select disjoint categories. CUDA events
measure stream elapsed time (including dispatch gaps), never kernel busy time.
"""
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
import time

_ACTIVE = ContextVar('training_profile', default=None)


def span(name):
    collector = _ACTIVE.get()
    return nullcontext() if collector is None else collector.span(name)


def measured(name, operation):
    with span(name):
        return operation()


def transfer(name, tensor, device):
    with span("h2d." + name):
        return tensor.to(device)


def sampled(indices, games):
    collector = _ACTIVE.get()
    if collector is not None and collector.capture_positions:
        identities = {id(g): g.get('game_id', str(i)) for i, g in enumerate(games)}
        collector.positions.append([(identities[id(g)], i) for g, i in indices])


class Collector:
    def __init__(self, *, cuda=False, records=False, timings=True, capture_positions=False, nvtx=False):
        self.nvtx = nvtx
        self.stack = []
        self.capture_positions = capture_positions
        self.timings = timings
        self.cuda = cuda
        self.records = records
        self.rows = []
        self.positions = []

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def span(self, name):
        return self._span(name) if self.timings else nullcontext()

    @contextmanager
    def _span(self, name):
        import torch
        if name.startswith('validate.') and name not in ('validate.pre', 'validate.post'):
            parent = next((n for n in reversed(self.stack) if n in ('validate.pre', 'validate.post')), None)
            if parent is not None:
                name = parent + '.' + name.split('.',1)[1]
        context = torch.profiler.record_function(name) if self.records else nullcontext()
        # Only coarse CUDA spans: per-parameter checks would add thousands of events.
        events = self.cuda and (not name.startswith(('validate.', 'io.')) or name in ('validate.pre','validate.post'))
        start = torch.cuda.Event(enable_timing=True) if events else None
        end = torch.cuda.Event(enable_timing=True) if events else None
        with context:
            self.stack.append(name)
            if self.nvtx:
                torch.cuda.nvtx.range_push(name)
            if start is not None:
                start.record()
            begin = time.perf_counter()
            try:
                yield
            finally:
                wall = (time.perf_counter() - begin) * 1000
                if end is not None:
                    end.record()
                self.rows.append((name, wall, start, end))
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
                self.stack.pop()

    def finish(self):
        # Called after the complete bounded pass, never between update stages.
        if self.cuda:
            import torch
            torch.cuda.synchronize()
        result = {}
        for name, wall, start, end in self.rows:
            row = result.setdefault(name, {'wall_ms': 0., 'stream_ms': 0., 'calls': 0})
            row['wall_ms'] += wall
            row['calls'] += 1
            if start is not None:
                row['stream_ms'] += start.elapsed_time(end)
        return result
