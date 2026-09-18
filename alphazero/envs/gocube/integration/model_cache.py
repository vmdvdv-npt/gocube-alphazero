"""Bounded cache for loaded Golden serving models.

The cache owns no game or position state.  It only reuses immutable loaded-model
objects and coalesces concurrent first loads of the same model identity.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable, Hashable


@dataclass
class _LoadFlight:
    event: Event
    value: object | None = None
    error: BaseException | None = None


class BoundedModelCache:
    """Thread-safe bounded LRU cache with single-flight loading per key."""

    def __init__(self, max_entries: int = 2):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("model cache size must be an integer >= 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, object] = OrderedDict()
        self._inflight: dict[Hashable, _LoadFlight] = {}
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> tuple[Hashable, ...]:
        """Return current LRU-to-MRU keys for diagnostics and tests."""

        with self._lock:
            return tuple(self._entries.keys())

    def get_or_load(self, key: Hashable, loader: Callable[[], object]) -> object:
        with self._lock:
            if key in self._entries:
                value = self._entries.pop(key)
                self._entries[key] = value
                return value

            flight = self._inflight.get(key)
            if flight is None:
                flight = _LoadFlight(event=Event())
                self._inflight[key] = flight
                owner = True
            else:
                owner = False

        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value

        try:
            value = loader()
        except BaseException as exc:
            with self._lock:
                flight.error = exc
                self._inflight.pop(key, None)
                flight.event.set()
            raise

        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            flight.value = value
            self._inflight.pop(key, None)
            flight.event.set()
        return value


__all__ = ["BoundedModelCache"]
