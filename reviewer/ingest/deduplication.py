from collections.abc import Callable
import time


class DeliveryStore:
    """Remembers claimed delivery IDs for `ttl_seconds`, in this process's memory.

    It's wiped on restart and not shared between workers (that needs Redis or a DB).
    `clock` is injectable so tests can move time forward without sleeping.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        if ttl_seconds <= 0:
            # A zero TTL would expire every ID at once: dedupe silently off.
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._seen: dict[str, float] = {}

    def _evict_expired(self, now: float) -> None:
        # Dicts keep insertion order and claim times only increase (monotonic
        # clock), so the oldest IDs come first: stop at the first unexpired one.
        # A released-then-reclaimed ID is re-inserted at the end, keeping the order.
        while self._seen:
            oldest_id, claimed_at = next(iter(self._seen.items()))
            if now - claimed_at < self.ttl_seconds:
                break
            del self._seen[oldest_id]

    def claim(self, delivery_id: str) -> bool:
        """Mark `delivery_id` as taken. True if it's new, False if it's a duplicate.

        Check and mark happen with no `await` in between, so two concurrent
        tasks can't both claim the same ID.
        """
        now = self._clock()
        self._evict_expired(now)

        if delivery_id in self._seen:
            return False

        self._seen[delivery_id] = now
        return True

    def release(self, delivery_id: str) -> None:
        """Forget `delivery_id` (after a failure) so a redelivery is processed."""
        self._seen.pop(delivery_id, None)

    def clear(self) -> None:
        self._seen.clear()

    def __contains__(self, delivery_id: object) -> bool:
        # Read-only check: doesn't evict, so an expired-but-not-yet-evicted ID
        # still shows as present until the next claim().
        return delivery_id in self._seen

    def __len__(self) -> int:
        return len(self._seen)
