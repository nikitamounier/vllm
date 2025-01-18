from functools import cached_property
from importlib.util import find_spec
from typing import Dict, Optional, Tuple

import torch
import torch.jit

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeStochasticBaseSampler)
from vllm.model_executor.layers.rejection_sampler import _multinomial

logger = init_logger(__name__)

if find_spec("flashinfer"):
    """
    Consider utilizing the FlashInfer rejection sampling kernel initially,
    as it employs a dedicated kernel rather than relying on 
    Torch tensor operations. This design choice helps to fuse operations, 
    reduce memory I/O, and consequently enhances performance.
    """
    from flashinfer.sampling import chain_speculative_sampling
else:
    chain_speculative_sampling = None


class MentoredSampler(SpecDecodeStochasticBaseSampler):
    """Apply mentored sampling as described in the paper.
    
    This sampler implements mentored decoding, which accelerates generation by using
    a smaller draft model while maintaining the target model's distribution quality.
    
    Key concepts:
    1. Draft model (p) proposes tokens that the target/mentor model (q) evaluates
    2. Instead of simple accept/reject based on q/p ratio, we:
       - Use acceptance probabilities r_i that maximize draft token acceptance
       - Subject to a KL divergence constraint: KL(q||π) ≤ D 
       - When rejecting, sample from an optimized fallback distribution s_i
    3. The resulting distribution π = p_i·r_i + s_i·(1-Σ p_j·r_j) approximates q
    
    The sampler solves for optimal r_i, s_i by:
    1. Binary search on parameter α that controls acceptance rates
    2. Setting r_i = min(1, q_i/(α·p_i)) 
    3. Computing s_i proportional to max(0, q_i/β - p_i)
    4. Ensuring KL(q||π) stays below target_kl threshold
    
    Optimization: If KL(q||p) ≤ target_kl for a token, we can accept immediately
    without computing r_i, s_i (quick accept path).
    """

    def __init__(
        self,
        target_kl: float = 0.2,  # Target KL divergence threshold
        kl_tolerance: float = 0.1,  # Tolerance for binary search
        max_binary_search_steps: int = 10,
        strict_mode: bool = False,
        use_flashinfer: Optional[bool] = None
    ):
        """Create a mentored sampler.

        Args:
            target_kl: The target KL divergence threshold D. Higher values allow more
                drift from the target distribution in exchange for higher acceptance rates.
                Typical values are 0.1-0.5.
            kl_tolerance: Tolerance for binary search on alpha parameter. Controls how
                precisely we match the target KL threshold.
            max_binary_search_steps: Maximum iterations for binary search on alpha.
                More steps = more precise but slower.
            strict_mode: Whether to perform shape/device/dtype checks during sampling.
            use_flashinfer: Whether to use FlashInfer kernel for sampling operations.
        """
        super().__init__(strict_mode=strict_mode)
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance 
        self.max_binary_search_steps = max_binary_search_steps
        if use_flashinfer is None:
            self.use_flashinfer = envs.VLLM_USE_FLASHINFER_SAMPLER and (
                chain_speculative_sampling is not None)
        else:
            self.use_flashinfer = use_flashinfer

        if self.use_flashinfer:
            logger.info("Use flashinfer for mentored sampling.")
        else:
            logger.info("Use pytorch for mentored sampling.")

    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,  # [batch_size, num_spec + 1, vocab]
        bonus_token_ids: torch.Tensor,  # [batch_size, num_bonus]
        draft_probs: torch.Tensor,  # [batch_size, num_spec, vocab] 
        draft_token_ids: torch.Tensor,  # [batch_size, num_spec]
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        """Sample token ids using mentored sampling.

        Args:
            target_with_bonus_probs: Target model probabilities including bonus token.
            bonus_token_ids: Bonus token ids to accept if all speculative tokens accepted.
            draft_probs: Draft model probabilities.
            draft_token_ids: Token ids sampled from draft probabilities.
            seeded_seqs: Dict mapping batch indices to torch generators for seeded sampling.

        Returns:
            output_token_ids: Sampled token ids, or -1 if token rejected.
                Shape = [batch_size, num_spec + num_bonus]
        """
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                         draft_token_ids, bonus_token_ids,
                                         draft_probs)

        batch_size, k, vocab_size = draft_probs.shape

        # Handle empty batch case
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=int)

        # Get uniform random samples for acceptance testing
        uniform_samples = self._create_uniform_samples(
            seeded_seqs, batch_size, k, draft_probs.device)

        # Compute acceptance probabilities and fallback distribution
        accepted, recovered_token_ids = self._batch_mentored_sampling(
            target_with_bonus_probs[:, :-1],  # Remove bonus token probs
            draft_probs,
            draft_token_ids,
            uniform_samples
        )

        # Create output combining accepted and recovered tokens
        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids, 
            draft_token_ids,
            bonus_token_ids
        )

        return output_token_ids

    def _batch_mentored_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        uniform_samples: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform mentored sampling on each sequence.

        Returns:
            Tuple containing:
            - Boolean tensor of which tokens are accepted [batch_size, k]
            - Token ids sampled from fallback distribution [batch_size, k]
        """
        batch_size, k, vocab_size = draft_probs.shape

        # Quick path: if KL(target||draft) <= target_kl, accept all tokens
        kl_div = torch.sum(target_probs * torch.log(target_probs / draft_probs), dim=-1)
        quick_accept = kl_div <= self.target_kl
        
        # For sequences requiring full mentored sampling:
        # 1. Sort q/p ratios
        q_over_p = target_probs / draft_probs
        sorted_ratios, sort_indices = torch.sort(q_over_p, dim=-1)
        sorted_target = torch.gather(target_probs, -1, sort_indices)
        sorted_draft = torch.gather(draft_probs, -1, sort_indices)

        # 2. Binary search for alpha
        alpha = self._binary_search_alpha(
            sorted_target, sorted_draft, sorted_ratios
        )

        # 3. Compute acceptance probabilities r_i
        acceptance_probs = torch.minimum(
            target_probs / (alpha.unsqueeze(-1) * draft_probs),
            torch.ones_like(draft_probs)
        )

        # 4. Accept/reject based on uniform samples
        batch_indices = torch.arange(batch_size, device=target_probs.device)[:, None]
        token_indices = torch.arange(k, device=target_probs.device)
        selected_acceptance_probs = acceptance_probs[
            batch_indices, token_indices, draft_token_ids
        ]
        accepted = uniform_samples < selected_acceptance_probs

        # Override with quick accept results
        accepted[quick_accept] = True

        # 5. Compute fallback distribution s_i
        fallback_probs = self._get_fallback_distribution(
            target_probs, draft_probs, alpha
        )

        # 6. Sample from fallback distribution when tokens rejected
        fallback_token_ids = _multinomial(
            fallback_probs.reshape(-1, vocab_size),
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs if seeded_seqs else {}
        ).reshape(batch_size, k)

        return accepted, fallback_token_ids

    def _binary_search_alpha(
        self,
        sorted_target: torch.Tensor,  # [batch_size, k, vocab]
        sorted_draft: torch.Tensor,   # [batch_size, k, vocab]
        sorted_ratios: torch.Tensor,  # [batch_size, k, vocab]
    ) -> torch.Tensor:  # [batch_size, k]
        """Binary search for alpha parameter that achieves target KL divergence."""
        batch_size, k, vocab_size = sorted_target.shape
        
        # Initialize search bounds
        alpha_min = torch.zeros(batch_size, k, device=sorted_target.device)
        alpha_max = torch.ones(batch_size, k, device=sorted_target.device)
        alpha = 0.5 * torch.ones_like(alpha_min)
        
        for _ in range(self.max_binary_search_steps):
            # Compute acceptance region
            accept_mask = sorted_ratios <= alpha.unsqueeze(-1)
            
            # Compute KL divergence
            r = torch.minimum(sorted_ratios / alpha.unsqueeze(-1), 
                            torch.ones_like(sorted_ratios))
            kl = torch.sum(sorted_target * torch.log(sorted_target / (r * sorted_draft)), dim=-1)
            
            # Update search bounds
            too_low = kl < self.target_kl * (1 - self.kl_tolerance)
            too_high = kl > self.target_kl * (1 + self.kl_tolerance)
            just_right = ~(too_low | too_high)
            
            alpha_max = torch.where(too_high, alpha, alpha_max)
            alpha_min = torch.where(too_low, alpha, alpha_min)
            alpha = torch.where(just_right, alpha, 0.5 * (alpha_min + alpha_max))
            
            if just_right.all():
                break
                
        return alpha

    def _get_fallback_distribution(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab]
        draft_probs: torch.Tensor,   # [batch_size, k, vocab]
        alpha: torch.Tensor,         # [batch_size, k]
    ) -> torch.Tensor:  # [batch_size, k, vocab]
        """Compute fallback distribution s_i for rejected tokens."""
        # Compute total acceptance probability R
        r = torch.minimum(
            target_probs / (alpha.unsqueeze(-1) * draft_probs),
            torch.ones_like(target_probs)
        )
        R = torch.sum(draft_probs * r, dim=-1, keepdim=True)
        
        # Compute beta for normalization
        beta = torch.sum(
            torch.maximum(target_probs - alpha.unsqueeze(-1) * draft_probs,
                         torch.zeros_like(target_probs)),
            dim=-1, keepdim=True
        )
        
        # Compute fallback distribution
        s = torch.maximum(
            target_probs - alpha.unsqueeze(-1) * draft_probs,
            torch.zeros_like(target_probs)
        ) / ((1 - R) * beta)
        
        return s
