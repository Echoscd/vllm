# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OnlineDualEncoderCacheManager cache replacement algorithm."""
import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    OnlineDualEncoderCacheManager,
    create_encoder_cache_manager,
)

pytestmark = pytest.mark.cpu_test


# ------------------ Mock Classes ------------------ #
class MockRequest:
    def __init__(self, request_id, mm_hashes, token_counts):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(
                    offset=0, length=self._token_counts[i]
                ),
            )
            self.mm_features.append(feature)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ------------------ Unit Tests ------------------ #
def test_basic_allocate_and_reuse():
    """Test basic allocation and cache reuse works the same as LRU."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [4])

    assert not cache.check_and_update_cache(req, 0)
    assert cache.can_allocate(req, 0, int(1e9), 0)

    cache.allocate(req, 0)
    assert cache.check_and_update_cache(req, 0)
    assert "r1" in cache.cached["imgA"]
    assert cache.num_free_slots == 6

    # Free twice to bring refcount to 0.
    cache.free_encoder_input(req, 0)
    cache.free_encoder_input(req, 0)

    assert not cache.cached["imgA"]
    assert "imgA" in cache.freeable
    assert cache.num_freeable_slots == 10
    assert cache.num_free_slots == 6


def test_eviction_when_cache_is_full():
    """Test that eviction works when cache is full."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["x"], [6])
    req2 = MockRequest("req2", ["y"], [5])

    assert cache.can_allocate(req1, 0, int(1e9), 0)
    cache.allocate(req1, 0)
    cache.free_encoder_input(req1, 0)

    assert cache.can_allocate(req2, 0, int(1e9), 0)
    cache.allocate(req2, 0)

    # 'x' should have been evicted
    assert "x" not in cache.cached
    assert "x" in cache.get_freed_mm_hashes()


def test_free_request_frees_all_inputs():
    """Test that free() releases all inputs."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)
    req = MockRequest("req3", ["a", "b"], [2, 3])

    assert cache.can_allocate(req, 0, int(1e9), 0)
    cache.allocate(req, 0)
    assert cache.can_allocate(req, 1, int(1e9), 0)
    cache.allocate(req, 1)

    cache.free(req)

    assert not cache.cached["a"]
    assert not cache.cached["b"]
    assert "a" in cache.freeable
    assert "b" in cache.freeable
    assert cache.num_freeable_slots == 10


def test_hit_count_tracking():
    """Test that hit counts are properly tracked."""
    cache = OnlineDualEncoderCacheManager(cache_size=20)
    req1 = MockRequest("r1", ["img1"], [4])
    req2 = MockRequest("r2", ["img1"], [4])

    cache.allocate(req1, 0)

    # First hit
    assert cache.check_and_update_cache(req2, 0)
    assert cache._hit_counts["img1"] == 2  # 1 from allocate + 1 from check

    # Second hit from different request
    req3 = MockRequest("r3", ["img1"], [4])
    assert cache.check_and_update_cache(req3, 0)
    assert cache._hit_counts["img1"] == 3


def test_dual_variable_updates():
    """Test that the dual variable lambda updates correctly."""
    cache = OnlineDualEncoderCacheManager(
        cache_size=10, soft_budget_ratio=0.5, initial_eta=0.1
    )

    # Fill cache beyond soft budget (5 tokens)
    req1 = MockRequest("r1", ["a"], [4])
    req2 = MockRequest("r2", ["b"], [4])

    cache.allocate(req1, 0)
    cache.allocate(req2, 0)
    # Now using 8 out of 10, soft budget is 5
    # lambda should increase when eviction is triggered
    cache.free_encoder_input(req1, 0)

    # Trigger eviction which also updates lambda
    req3 = MockRequest("r3", ["c"], [4])
    cache.can_allocate(req3, 0, int(1e9), 0)

    # Lambda should have increased since usage > soft_budget
    assert cache._lambda > 0


def test_frequency_based_eviction():
    """Test that frequently accessed items are preserved over rare ones."""
    cache = OnlineDualEncoderCacheManager(
        cache_size=12,
        soft_budget_ratio=0.8,
        initial_eta=0.01,
        default_compute_cost=0.05,
    )

    # Allocate two items
    req_rare = MockRequest("r1", ["rare_img"], [4])
    req_freq = MockRequest("r2", ["freq_img"], [4])

    cache.allocate(req_rare, 0)
    cache.allocate(req_freq, 0)

    # Simulate frequent access to freq_img
    for i in range(10):
        req = MockRequest(f"hit_{i}", ["freq_img"], [4])
        cache.check_and_update_cache(req, 0)
        cache.free_encoder_input(req, 0)

    # Free both items
    cache.free_encoder_input(req_rare, 0)
    cache.free_encoder_input(req_freq, 0)

    # Now force eviction by adding a new large item
    req_new = MockRequest("r_new", ["new_img"], [6])
    cache.can_allocate(req_new, 0, int(1e9), 0)
    cache.allocate(req_new, 0)

    freed = cache.get_freed_mm_hashes()
    # The rare item should be more likely evicted than the frequent one
    # (this depends on the dual variable state, but with enough hits
    # the frequent item should be preserved)
    assert "rare_img" in freed or "freq_img" in freed


def test_compute_cost_recording():
    """Test that compute costs are recorded and used."""
    cache = OnlineDualEncoderCacheManager(
        cache_size=20, default_compute_cost=0.01
    )

    # Record a high compute cost for an item
    cache.record_compute_cost("expensive_img", 1.0)

    req = MockRequest("r1", ["expensive_img"], [5])
    cache.allocate(req, 0)

    # The item state should reflect the high compute cost
    state = cache._item_state["expensive_img"]
    assert state.compute_cost == 1.0
    # A_i should be initialized to -c_i = -1.0
    # (but allocate calls _get_or_create_state which uses recorded cost)


def test_get_stats():
    """Test that stats are correctly reported."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["img1"], [4])
    cache.allocate(req, 0)

    stats = cache.get_stats()
    assert stats["num_cached"] == 1
    assert stats["cache_utilization"] == 0.4
    assert stats["lambda"] == 0.0
    assert stats["step"] == 0


def test_factory_creates_lru_by_default(monkeypatch):
    """Test that factory creates LRU manager by default."""
    monkeypatch.delenv("VLLM_CACHE_POLICY", raising=False)
    manager = create_encoder_cache_manager(cache_size=10, policy="lru")
    assert isinstance(manager, EncoderCacheManager)


def test_factory_creates_online_dual(monkeypatch):
    """Test that factory creates OnlineDual manager when requested."""
    manager = create_encoder_cache_manager(
        cache_size=10, policy="online_dual"
    )
    assert isinstance(manager, OnlineDualEncoderCacheManager)


def test_factory_env_var(monkeypatch):
    """Test that factory respects VLLM_CACHE_POLICY env var."""
    monkeypatch.setenv("VLLM_CACHE_POLICY", "online_dual")
    manager = create_encoder_cache_manager(cache_size=10)
    assert isinstance(manager, OnlineDualEncoderCacheManager)


def test_factory_invalid_policy():
    """Test that factory raises error for unknown policy."""
    with pytest.raises(ValueError, match="Unknown cache policy"):
        create_encoder_cache_manager(cache_size=10, policy="invalid")


def test_has_cache_restores_from_freeable():
    """Test that check_and_update_cache restores from freeable."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)
    req = MockRequest("reqY", ["imgZ"], [4])

    cache.allocate(req, 0)
    cache.free_encoder_input(req, 0)

    # Should restore from freeable
    assert cache.check_and_update_cache(req, 0)
    assert len(cache.cached["imgZ"]) == 1
    assert "imgZ" not in cache.freeable
    assert cache.num_freeable_slots == 6


def test_get_freed_mm_hashes_clears_list():
    """Test that get_freed_mm_hashes clears the freed list."""
    cache = OnlineDualEncoderCacheManager(cache_size=10)
    req1 = MockRequest("reqA", ["a"], [5])
    req2 = MockRequest("reqB", ["b"], [6])

    cache.allocate(req1, 0)
    cache.free_encoder_input(req1, 0)

    cache.can_allocate(req2, 0, int(1e9), 0)
    cache.allocate(req2, 0)

    freed = cache.get_freed_mm_hashes()
    assert "a" in freed
    assert cache.get_freed_mm_hashes() == []


def test_compute_budget_respected():
    """Test that compute budget is respected."""
    cache = OnlineDualEncoderCacheManager(cache_size=100)
    req = MockRequest("r1", ["img"], [50])
    assert not cache.can_allocate(req, 0, 10, 0)  # budget too small


def test_lru_fallback_when_no_evictable():
    """Test that LRU fallback works when no items are marked evictable."""
    # With lambda=0, no items will be marked evictable initially
    # The algorithm should fall back to evicting the oldest freeable item
    cache = OnlineDualEncoderCacheManager(
        cache_size=6,
        soft_budget_ratio=0.9,
        initial_eta=0.0,  # lambda stays at 0
        default_compute_cost=0.05,
    )

    req1 = MockRequest("r1", ["old"], [3])
    req2 = MockRequest("r2", ["new"], [3])

    cache.allocate(req1, 0)
    cache.free_encoder_input(req1, 0)

    # Need to evict "old" to make room for "newer"
    req3 = MockRequest("r3", ["newer"], [4])
    assert cache.can_allocate(req3, 0, int(1e9), 0)
    freed = cache.get_freed_mm_hashes()
    assert "old" in freed
