import pytest

from reviewer.ingest.deduplication import DeliveryStore

TTL = 100.0


class FakeClock:
    """A clock the test moves by hand: `clock.now = 99.9`, no sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock) -> DeliveryStore:
    return DeliveryStore(ttl_seconds=TTL, clock=clock)


class TestClaim:
    """`claim(id)`: True the first time, False for a duplicate (atomic check-and-mark)."""

    def test_new_id_is_claimed(self, store):
        assert store.claim("a") is True
        assert "a" in store

    def test_duplicate_is_refused(self, store):
        store.claim("a")

        assert store.claim("a") is False

    def test_ids_are_independent(self, store):
        store.claim("a")

        assert store.claim("b") is True
        assert len(store) == 2


class TestRelease:
    """`release(id)` after a failure lets a redelivery of that ID run again."""

    def test_released_id_can_be_claimed_again(self, store):
        store.claim("a")
        store.release("a")

        assert "a" not in store
        assert store.claim("a") is True

    def test_releasing_an_unknown_id_is_a_no_op(self, store):
        store.release("never-claimed")  # must not raise

        assert len(store) == 0


class TestTTL:
    """IDs are remembered for exactly `ttl_seconds` (expiry is `>=`)."""

    def test_still_a_duplicate_just_before_the_ttl(self, store, clock):
        store.claim("a")
        clock.now = TTL - 0.001

        assert store.claim("a") is False

    def test_claimable_again_at_exactly_the_ttl(self, store, clock):
        store.claim("a")
        clock.now = TTL

        assert store.claim("a") is True

    def test_expired_ids_are_removed(self, store, clock):
        # Memory stays bounded: old IDs are dropped, not just ignored.
        store.claim("a")
        store.claim("b")
        clock.now = TTL
        store.claim("c")

        assert len(store) == 1
        assert "a" not in store and "b" not in store

    def test_eviction_keeps_unexpired_ids(self, store, clock):
        # The early-stop eviction relies on insertion order = claim-time order.
        store.claim("old")      # t=0
        clock.now = 50.0
        store.claim("young")    # t=50
        clock.now = TTL         # "old" expires, "young" has 50 s left
        store.claim("trigger")

        assert "old" not in store
        assert "young" in store

    def test_reclaimed_id_gets_a_fresh_ttl(self, store, clock):
        # release + claim re-inserts at the END with the new time, so the early
        # stop still sees the oldest first and doesn't evict it too soon.
        store.claim("a")         # t=0
        clock.now = 10.0
        store.claim("b")         # t=10
        store.release("a")
        clock.now = 20.0
        store.claim("a")         # t=20, now after "b"
        clock.now = TTL + 10.0   # "b" expires (t=10); "a" (t=20) has 10 s left
        store.claim("trigger")

        assert "b" not in store
        assert "a" in store


class TestConstruction:
    @pytest.mark.parametrize("ttl", [0, -1])
    def test_non_positive_ttl_is_rejected(self, ttl):
        # A zero TTL would expire every ID at once: dedupe silently off.
        with pytest.raises(ValueError, match="ttl_seconds"):
            DeliveryStore(ttl_seconds=ttl)

    def test_clear_forgets_everything(self, store):
        store.claim("a")
        store.claim("b")
        store.clear()

        assert len(store) == 0
        assert store.claim("a") is True
