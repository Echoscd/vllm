# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    tokens from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder tokens.
        num_free_slots: Current available cache capacity in encoder tokens.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder tokens).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_tokens) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_tokens of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_tokens = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_tokens

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_tokens_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True.
        Older entries are evicted first.

        Returns False only if the requested number of tokens exceeds both
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder tokens allowed to be
                computed when this method is invoked.
            num_tokens_to_schedule: Number of tokens already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)

        # Not enough compute budget
        if num_tokens > encoder_compute_budget:
            return False

        num_tokens += num_tokens_to_schedule

        # Enough free slots
        if num_tokens <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_tokens > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_tokens > self.num_free_slots:
            mm_hash, num_free_token = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_token
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_tokens = request.get_num_encoder_tokens(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_tokens
        assert self.num_freeable_slots >= num_encoder_tokens

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_tokens
        self.num_freeable_slots -= num_encoder_tokens

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder tokens for that input.

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_tokens = request.get_num_encoder_tokens(input_id)
            self.freeable[mm_hash] = num_tokens
            self.num_freeable_slots += num_tokens

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.
        The data stays in memory until eviction is triggered by a future
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which
            encoder outputs can be removed from their caches. The internal
            list is cleared after this call.
        """
        freed = self.freed
        self.freed = []
        return freed


@dataclass
class _OnlineDualItemState:
    """Per-item state for the online dual-variable cache algorithm."""
    mm_hash: str
    num_tokens: int  # m_i: memory cost in tokens
    compute_cost: float  # c_i: encoder compute cost (seconds)
    dual_accum: float = 0.0  # A_i: accumulated dual variable
    evictable: bool = False  # whether this item is marked evictable
    last_step_updated: int = 0  # last step when A_i was updated


class OnlineDualEncoderCacheManager:
    """Cache manager using online dual-variable optimization for eviction.

    This algorithm replaces LRU with a principled online learning approach
    that considers both memory cost (m_i) and compute cost (c_i) of each
    cached item, along with request frequency patterns.

    Algorithm overview:
    1. Each item i has state: A_i (dual accumulator), hit count, evictability
    2. When item is evicted: reset A_i = -c_i (penalty for recomputation)
    3. On each request arrival (step t), for all cached items:
       - A_i += m_i * lambda_t  (charge for memory usage, proportional to
         the number of steps since last update)
       - evictable = (A_i + m_i * lambda_t * n_{t,i} > 0)
         where n_{t,i} = t / hit_count_i (inverse frequency)
    4. On cache miss when over soft budget: evict items marked evictable
    5. Update lambda: lambda_{t+1} = lambda_t + eta_t * (U_t - B)
       where U_t = used cache, B = soft budget (fraction of real capacity)

    Args:
        cache_size: Real cache capacity in encoder tokens.
        soft_budget_ratio: Fraction of cache_size used as soft budget B.
        initial_eta: Initial learning rate for lambda updates.
        default_compute_cost: Default c_i when no timing data is available.
    """

    def __init__(
        self,
        cache_size: int,
        soft_budget_ratio: float = 0.9,
        initial_eta: float = 0.001,
        default_compute_cost: float = 0.05,
    ):
        self.cache_size = cache_size
        self.soft_budget = int(cache_size * soft_budget_ratio)
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash => set of request IDs referencing this item
        self.cached: dict[str, set[str]] = {}

        # mm_hash => num_tokens (for unreferenced items eligible for eviction)
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

        # Online dual-variable state
        self._lambda: float = 0.0  # dual variable (price of memory)
        self._eta: float = initial_eta  # learning rate
        self._default_compute_cost = default_compute_cost
        self._step: int = 0  # global step counter

        # Per-item state: mm_hash => _OnlineDualItemState
        self._item_state: dict[str, _OnlineDualItemState] = {}

        # Global hit tracking: mm_hash => total hit count
        self._hit_counts: dict[str, int] = {}

        # Compute cost tracking: mm_hash => measured compute time (seconds)
        self._compute_costs: dict[str, float] = {}

    def _get_or_create_state(
        self, mm_hash: str, num_tokens: int
    ) -> _OnlineDualItemState:
        """Get or create per-item state for an mm_hash."""
        if mm_hash not in self._item_state:
            c_i = self._compute_costs.get(
                mm_hash, self._default_compute_cost
            )
            self._item_state[mm_hash] = _OnlineDualItemState(
                mm_hash=mm_hash,
                num_tokens=num_tokens,
                compute_cost=c_i,
                dual_accum=-c_i,  # Initialize A_i = -c_i
                last_step_updated=self._step,
            )
        return self._item_state[mm_hash]

    def _step_and_update(self) -> None:
        """Advance step counter and update dual state for all cached items.

        This implements the per-step update of the algorithm:
        1. Advance step counter
        2. Update lambda based on cache pressure
        3. Accumulate A_i for all cached items (lazy batch update)
        4. Update evictability for freeable items
        """
        self._step += 1

        # Update lambda: lambda_{t+1} = max(0, lambda_t + eta * (U_t - B))
        used = self.cache_size - self.num_free_slots
        self._lambda = max(
            0.0, self._lambda + self._eta * (used - self.soft_budget)
        )

        # Update A_i and evictability for freeable items
        # (items with references are not evictable anyway)
        for mm_hash, num_tokens in self.freeable.items():
            state = self._get_or_create_state(mm_hash, num_tokens)
            m_i = state.num_tokens

            # Catch up: accumulate for all steps since last update
            steps_elapsed = self._step - state.last_step_updated
            state.dual_accum += m_i * self._lambda * steps_elapsed
            state.last_step_updated = self._step

            # Compute n_{t,i} = t / hit_count (inverse frequency estimate)
            hit_count = max(self._hit_counts.get(mm_hash, 1), 1)
            n_ti = max(self._step, 1) / hit_count

            # Evictable if A_i + m_i * lambda * n_{t,i} > 0
            state.evictable = (
                state.dual_accum + m_i * self._lambda * n_ti > 0
            )

    def _select_eviction_victim(self) -> str | None:
        """Select which freeable item to evict based on the dual algorithm.

        Among evictable items, picks the one with the highest eviction score:
            score = A_i / c_i  (accumulated cost normalized by compute cost)
        This prefers evicting items that have accumulated lots of memory cost
        relative to their recomputation cost.

        Falls back to LRU (oldest freeable) if no evictable items exist.
        """
        best_hash = None
        best_score = float("-inf")

        for mm_hash in self.freeable:
            state = self._item_state.get(mm_hash)
            if state is not None and state.evictable:
                # Prefer items with high A_i / c_i
                score = state.dual_accum / max(state.compute_cost, 1e-9)
                if score > best_score:
                    best_score = score
                    best_hash = mm_hash

        if best_hash is not None:
            return best_hash

        # Fallback: evict oldest freeable item (LRU fallback)
        if self.freeable:
            return next(iter(self.freeable))
        return None

    def record_compute_cost(self, mm_hash: str, cost_seconds: float) -> None:
        """Record measured encoder compute time for an mm_hash.

        This should be called after encoder execution to provide accurate
        cost estimates for the eviction algorithm.
        """
        self._compute_costs[mm_hash] = cost_seconds
        if mm_hash in self._item_state:
            self._item_state[mm_hash].compute_cost = cost_seconds

    def check_and_update_cache(
        self, request: Request, input_id: int
    ) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        On each call, advances the step counter and updates the dual state
        for all cached items. On cache hit, increments hit count.
        """
        mm_hash = request.mm_features[input_id].identifier
        if mm_hash not in self.cached:
            # Step even on miss to keep lambda updated
            self._step_and_update()
            return False

        # Step and update dual state on every request
        self._step_and_update()

        # Record hit
        self._hit_counts[mm_hash] = self._hit_counts.get(mm_hash, 0) + 1

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_tokens = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_tokens
            # Mark as non-evictable since it's now referenced
            if mm_hash in self._item_state:
                self._item_state[mm_hash].evictable = False

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_tokens_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.

        Uses the online dual-variable algorithm for eviction decisions instead
        of simple LRU. The algorithm considers item compute cost, memory cost,
        and access frequency when choosing what to evict.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)

        # Not enough compute budget
        if num_tokens > encoder_compute_budget:
            return False

        num_tokens += num_tokens_to_schedule

        # Enough free slots
        if num_tokens <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_tokens > self.num_freeable_slots:
            return False

        # Evict using the online dual algorithm
        while num_tokens > self.num_free_slots:
            victim = self._select_eviction_victim()
            if victim is None:
                # No more items to evict
                return False

            num_free_token = self.freeable.pop(victim)
            del self.cached[victim]
            self.freed.append(victim)
            self.num_free_slots += num_free_token

            # Reset dual state for evicted item: A_i = -c_i
            if victim in self._item_state:
                state = self._item_state[victim]
                state.dual_accum = -state.compute_cost
                state.evictable = False
                state.last_step_updated = self._step

        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output."""
        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_tokens = request.get_num_encoder_tokens(input_id)

        assert self.num_free_slots >= num_encoder_tokens
        assert self.num_freeable_slots >= num_encoder_tokens

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_tokens
        self.num_freeable_slots -= num_encoder_tokens

        # Initialize item state if new
        self._get_or_create_state(mm_hash, num_encoder_tokens)
        # Record hit for newly allocated items
        self._hit_counts[mm_hash] = self._hit_counts.get(mm_hash, 0) + 1

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request."""
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input."""
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_tokens = request.get_num_encoder_tokens(input_id)
            self.freeable[mm_hash] = num_tokens
            self.num_freeable_slots += num_tokens

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*."""
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries."""
        freed = self.freed
        self.freed = []
        return freed

    def get_stats(self) -> dict:
        """Return algorithm statistics for benchmarking."""
        return {
            "lambda": self._lambda,
            "step": self._step,
            "eta": self._eta,
            "num_tracked_items": len(self._item_state),
            "num_cached": len(self.cached),
            "num_freeable": len(self.freeable),
            "cache_utilization": (
                (self.cache_size - self.num_free_slots) / self.cache_size
                if self.cache_size > 0
                else 0.0
            ),
        }


def create_encoder_cache_manager(
    cache_size: int,
    policy: str | None = None,
) -> EncoderCacheManager | OnlineDualEncoderCacheManager:
    """Factory function to create the appropriate encoder cache manager.

    Args:
        cache_size: Cache capacity in encoder tokens.
        policy: Cache replacement policy. Options:
            - "lru" (default): Original LRU-based eviction.
            - "online_dual": Online dual-variable optimization.
            Can also be set via VLLM_CACHE_POLICY environment variable.

    Returns:
        An encoder cache manager instance.
    """
    if policy is None:
        policy = envs.VLLM_CACHE_POLICY.lower()

    if policy == "online_dual":
        soft_ratio = envs.VLLM_CACHE_SOFT_BUDGET_RATIO
        eta = envs.VLLM_CACHE_ETA
        default_cost = envs.VLLM_CACHE_DEFAULT_COMPUTE_COST
        logger.info(
            "Using OnlineDual encoder cache policy "
            "(soft_budget_ratio=%.2f, eta=%.4f, default_cost=%.4f)",
            soft_ratio,
            eta,
            default_cost,
        )
        return OnlineDualEncoderCacheManager(
            cache_size=cache_size,
            soft_budget_ratio=soft_ratio,
            initial_eta=eta,
            default_compute_cost=default_cost,
        )
    elif policy == "lru":
        logger.info("Using LRU encoder cache policy")
        return EncoderCacheManager(cache_size=cache_size)
    else:
        raise ValueError(
            f"Unknown cache policy '{policy}'. Use 'lru' or 'online_dual'."
        )


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """
    if mm_registry.supports_multimodal_inputs(model_config):
        max_tokens_by_modality = mm_registry.get_max_tokens_per_item_by_modality(
            model_config
        )

        return compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

    return compute_text_encoder_budget(scheduler_config)


def compute_text_encoder_budget(scheduler_config: "SchedulerConfig") -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a text-only model.

    Args:
        scheduler_config: Scheduler configuration.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens
            in the input sequence.
    """
    # Currently text-only encoder-decoder models are not supported
    return 0, 0


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    max_tokens_by_modality: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        max_tokens_by_modality: The maximum number of tokens for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not max_tokens_by_modality:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(max_tokens_by_modality.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    # Allow overriding cache size via env var for benchmarking
    override = envs.VLLM_ENCODER_CACHE_SIZE
    if override > 0:
        if override < max_tokens_per_mm_item:
            logger.warning(
                "VLLM_ENCODER_CACHE_SIZE=%d is smaller than "
                "max_tokens_per_mm_item=%d. Clamping to %d.",
                override,
                max_tokens_per_mm_item,
                max_tokens_per_mm_item,
            )
            override = max_tokens_per_mm_item
        logger.info(
            "Overriding encoder_cache_size: %d -> %d (via VLLM_ENCODER_CACHE_SIZE)",
            encoder_cache_size,
            override,
        )
        encoder_cache_size = override

    return encoder_compute_budget, encoder_cache_size
