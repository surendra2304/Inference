"""Bounded in-process stores: the standard accumulator for this codebase.

Motivation (measured, not theoretical)
--------------------------------------
A 60-second soak at concurrency 20 (17,765 requests, 0 errors) grew the resident set of
the agent process from 77.9 MB to 143.2 MB and it did not come back down 3 seconds after
the load stopped (144.3 MB). Per-route attribution then measured the growth per 1,000
requests, with a ``gc.collect()`` before sampling so that ordinary garbage could not be
mistaken for a leak::

    nexus      +9.03 MB / 1k requests   retained +19.7 MB
    sentinel   +5.19 MB / 1k requests   retained +13.7 MB
    analytics  +2.42 MB / 1k requests   retained +10.4 MB
    instant    +0.50 MB / 1k requests   (cache-bounded)
    market     +0.20 MB / 1k requests   (no accumulator)
    stress     +0.02 MB / 1k requests   (no accumulator)
    assist     +0.02 MB / 1k requests   (no accumulator)

Every one of those was the same shape: a module-level singleton with a plain ``{}`` or
``[]`` that every request appended to and nothing ever removed (``nexus_intelligence.py``
``provenance_store``, ``sentinel_intelligence.py`` ``provenance_store``,
``outcome_learning.py`` ``outcome_records`` / ``strategy_bank``, and so on — the same
``self.x = []`` pattern appears in a dozen more modules). On a long-lived process that is
an unbounded leak, and the audit endpoints above are exactly the ones an operator leaves
running.

What this module provides
-------------------------
``BoundedStore``   — mapping with LRU eviction; ``evictions`` counts what was dropped.
``BoundedList``    — append-only ring buffer with the same accounting.
``BoundedSeries``  — numeric series (latency samples, cost samples) that keeps the last N
                     and, separately, the total count ever added.

Every store registers itself so tests can audit the whole process: ``audit_bounds()``
returns one row per live store with ``used``, ``max_entries`` and ``evicted``. The point
is that *eviction is not silent*: a store that has dropped entries says so, and callers
that serve "not found" for an evicted id can distinguish that from "never existed"
(``BoundedStore.evicted_marker``).

Honesty note: bounding a store changes observable behaviour — a trace older than the last
N requests is no longer retrievable. That is the intended trade (bounded memory over
unbounded history) and it must be *said*, not hidden, which is why ``describe()`` exists
and why the retrieving endpoints report eviction explicitly.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from typing import Any, Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")

#: Default retention for provenance-style stores (one entry per request).
DEFAULT_MAX_ENTRIES = 2048

#: Default retention for outcome/telemetry ledgers.
DEFAULT_MAX_RECORDS = 10_000


class BoundedStore(Generic[T]):
    """A dict with a hard ceiling and least-recently-used eviction.

    Reads refresh recency, so a hot key is not evicted by a cold flood. ``evictions``
    counts entries dropped for capacity; ``evicted_keys`` remembers the most recent of
    them (bounded itself) so callers can answer "was this dropped?" as opposed to "was
    this ever here?".
    """

    __slots__ = ("_data", "_max_entries", "_evicted_keys", "_lock", "name", "evictions", "insertions")

    #: How many recently evicted keys are remembered exactly. A lookup miss for one of
    #: them can be answered confidently ("expired"); for anything older the store cannot
    #: tell eviction from absence and must say so rather than pick one.
    EVICTED_KEY_MEMORY = 512

    def __init__(self, name: str, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.name = name
        self._max_entries = max_entries
        self._data: OrderedDict[str, T] = OrderedDict()
        self._evicted_keys: deque[str] = deque(maxlen=self.EVICTED_KEY_MEMORY)
        self._lock = threading.Lock()
        self.evictions = 0
        self.insertions = 0
        register(self)

    # -- mapping protocol ---------------------------------------------------
    # Keys are canonicalised to ``str``. A request id that reaches a store as an int must
    # map to the same slot as the same id passed as a string, otherwise ``was_evicted``
    # answers about a key nobody inserted.
    def __setitem__(self, key: str, value: T) -> None:
        key = str(key)
        with self._lock:
            if key in self._data:
                self._data[key] = value
                self._data.move_to_end(key)
            else:
                self._data[key] = value
                self.insertions += 1
            while len(self._data) > self._max_entries:
                dropped, _ = self._data.popitem(last=False)
                self._evicted_keys.append(dropped)
                self.evictions += 1

    def __getitem__(self, key: str) -> T:
        key = str(key)
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def setdefault(self, key: str, default: T) -> T:
        key = str(key)
        with self._lock:
            if key not in self._data:
                self[key] = default
            return self[key]

    def get(self, key: str, default: T | None = None) -> T | None:
        key = str(key)
        with self._lock:
            if key not in self._data:
                return default
            return self._data[key]

    def pop(self, key: str, default: T | None = None) -> T | None:
        key = str(key)
        with self._lock:
            return self._data.pop(key, default)

    def __delitem__(self, key: str) -> None:
        key = str(key)
        with self._lock:
            del self._data[key]

    def __contains__(self, key: object) -> bool:
        key = str(key)
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate keys, snapshot under the lock.

        Without this, a mapping-style store is not iterable at all: ``for k in store`` and
        ``k in store`` inside a comprehension fall back to the *sequence* protocol and call
        ``__getitem__(0)``, ``__getitem__(1)``... which raised ``KeyError: '0'`` for the
        first missing key instead of iterating (found by a test that did
        ``any("stale-" in k for k in mgr.dedup_cache)``).
        """
        with self._lock:
            return iter(list(self._data.keys()))

    def to_dict(self) -> dict[str, T]:
        """A plain dict snapshot, for callers that must serialize."""
        with self._lock:
            return dict(self._data)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> list[T]:
        with self._lock:
            return list(self._data.values())

    def items(self) -> list[tuple[str, T]]:
        with self._lock:
            return list(self._data.items())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # -- diagnostics --------------------------------------------------------
    @property
    def max_entries(self) -> int:
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """Adjust the ceiling, trimming immediately so ``len <= max_entries`` always holds.

        A settable ceiling that did not trim would let the store report a bound it does not
        actually enforce, which is the defect class this module exists to prevent.
        """
        if value <= 0:
            raise ValueError("max_entries must be positive")
        with self._lock:
            self._max_entries = value
            while len(self._data) > self._max_entries:
                dropped, _ = self._data.popitem(last=False)
                self._evicted_keys.append(dropped)
                self.evictions += 1

    def was_evicted(self, key: str) -> bool:
        """True when ``key`` was dropped for capacity (so it may have existed)."""
        key = str(key)
        with self._lock:
            return key in self._evicted_keys

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "kind": "BoundedStore",
                "used": len(self._data),
                "max_entries": self._max_entries,
                "insertions": self.insertions,
                "evicted": self.evictions,
                "within_bound": len(self._data) <= self._max_entries,
            }


class BoundedList(Generic[T]):
    """A list with a hard ceiling that keeps the most recent entries.

    Iteration order is oldest-to-newest, matching ``list``. ``dropped`` counts entries
    that fell off the back, so a consumer that computes a rate over "all records" can
    state how many it never saw.
    """

    __slots__ = ("_data", "_lock", "name", "dropped", "insertions")

    def __init__(self, name: str, max_entries: int = DEFAULT_MAX_RECORDS,
                 initial: Iterable[T] | None = None) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.name = name
        self._data: deque[T] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self.dropped = 0
        self.insertions = 0
        if initial:
            for item in initial:
                self.append(item)
        register(self)

    def append(self, item: T) -> None:
        with self._lock:
            was_full = len(self._data) == self._data.maxlen
            self._data.append(item)
            self.insertions += 1
            if was_full:
                self.dropped += 1

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[T]:
        with self._lock:
            return iter(list(self._data))

    def __getitem__(self, index: int | slice) -> Any:
        with self._lock:
            data = list(self._data)
        return data[index]

    def __contains__(self, item: object) -> bool:
        with self._lock:
            return item in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def to_list(self) -> list[T]:
        with self._lock:
            return list(self._data)

    @property
    def max_entries(self) -> int:
        return self._data.maxlen or 0

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "kind": "BoundedList",
                "used": len(self._data),
                "max_entries": self._data.maxlen,
                "insertions": self.insertions,
                "evicted": self.dropped,
                "within_bound": len(self._data) <= (self._data.maxlen or 0),
            }


class BoundedSeries:
    """A bounded numeric series that also remembers how many samples it ever saw.

    Percentiles must be computed from the retained window; that is a limitation worth
    stating in the payload rather than presenting a window statistic as an all-time one.
    ``count_total`` is what an honesty audit needs: "p99 over the last 1024 samples, of
    41,233 observed".
    """

    __slots__ = ("_values", "_lock", "name", "dropped", "count_total")

    def __init__(self, name: str, max_samples: int = 1024) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.name = name
        self._values: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self.dropped = 0
        self.count_total = 0
        register(self)

    def add(self, value: float) -> None:
        with self._lock:
            if len(self._values) == self._values.maxlen:
                self.dropped += 1
            self._values.append(value)
            self.count_total += 1

    def values(self) -> list[float]:
        with self._lock:
            return list(self._values)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    @property
    def max_entries(self) -> int:
        return self._values.maxlen or 0

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "kind": "BoundedSeries",
                "used": len(self._values),
                "max_entries": self._values.maxlen,
                "insertions": self.count_total,
                "evicted": self.dropped,
                "within_bound": len(self._values) <= (self._values.maxlen or 0),
            }


# ---------------------------------------------------------------------------
# Process-wide registry: lets a test assert that nothing in the process grows
# without a ceiling, including stores added later by someone else.
# ---------------------------------------------------------------------------
_REGISTRY: "list[BoundedStore[Any] | BoundedList[Any] | BoundedSeries]" = []
_REGISTRY_LOCK = threading.Lock()


def register(store: "BoundedStore[Any] | BoundedList[Any] | BoundedSeries") -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.append(store)


def audit_bounds() -> dict[str, Any]:
    """One row per live bounded store, plus a verdict.

    ``within_bound`` is False if any store exceeds its ceiling, which would mean the
    bounding logic itself is broken (not that the ceiling is too small).
    """
    with _REGISTRY_LOCK:
        stores = list(_REGISTRY)
    rows = [store.describe() for store in stores]
    overflowing = [row["name"] for row in rows if not row["within_bound"]]
    evicting = [row["name"] for row in rows if row["evicted"]]
    return {
        "stores": sorted(rows, key=lambda r: r["name"]),
        "store_count": len(rows),
        "within_bound": not overflowing,
        "overflowing": overflowing,
        "evicting_stores": evicting,
        "total_evictions": sum(row["evicted"] for row in rows),
    }


def registry_snapshot() -> list[dict[str, Any]]:
    with _REGISTRY_LOCK:
        stores = list(_REGISTRY)
    return [store.describe() for store in stores]


def missing_entry_detail(store: "BoundedStore[Any] | BoundedList[Any]",
                         key: str, what: str = "record") -> str:
    """Build the detail string for a lookup miss, saying *which kind* of miss it is.

    A retention window makes "never recorded" and "recorded, then evicted for capacity"
    different facts, and an audit endpoint that answers both with "not found" tells the
    caller their request never happened, which may be false.

    Three cases, in order of confidence:

    * the key is in the store's recent-eviction memory -> it *was* recorded, now expired;
    * the store has evicted entries it can no longer name -> the miss is *ambiguous* and
      the message says both possibilities rather than asserting one. (This case is real:
      driving 2,601 unique nexus requests evicted 1,106 entries, so the first request's
      key was long out of a 64-key memory and the endpoint answered "never recorded" for
      a request that had succeeded with HTTP 200.);
    * nothing has ever been evicted -> the key genuinely was never recorded.
    """
    desc = store.describe()
    if isinstance(store, BoundedStore) and store.was_evicted(key):
        return (
            f"The {what} for '{key}' has expired from the retention window: this service "
            f"keeps the most recent {desc['max_entries']} entries and has evicted "
            f"{desc['evicted']} in total. Re-run the request to regenerate it."
        )
    if desc["evicted"]:
        return (
            f"No {what} is retained for '{key}': it was either never recorded, or it has "
            f"been evicted from the retention window (this service keeps the most recent "
            f"{desc['max_entries']} entries and has evicted {desc['evicted']}). Re-run the "
            f"request to regenerate it."
        )
    return (
        f"No {what} was recorded for '{key}'. "
        f"({desc['used']} of {desc['max_entries']} retention slots are in use, none evicted.)"
    )
