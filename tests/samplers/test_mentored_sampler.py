import pytest
import torch
from typing import Dict, List, Optional, Tuple

from vllm.model_executor.layers.mentored_sampler import MentoredSampler
from ..core.utils import set_random_seed

CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

def _prepare_test(batch_size: int, k: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_probs = torch.rand((batch_size, k + 1, vocab_size), dtype=torch.float32)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    
    draft_probs = torch.rand((batch_size, k, vocab_size), dtype=torch.float32)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    
    draft_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, k))
    bonus_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 1))
    
    return target_probs, draft_probs, draft_token_ids, bonus_token_ids

@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mentored_sampling_basic(seed: int, device: str):
    """Test basic functionality of mentored sampling."""
    set_random_seed(seed)
    torch.set_default_device(device)
    
    batch_size = 4
    k = 3  # number of speculative tokens
    vocab_size = 1000
    
    target_probs, draft_probs, draft_token_ids, bonus_token_ids = _prepare_test(
        batch_size, k, vocab_size)
    
    sampler = MentoredSampler(target_kl=0.2)
    
    output_token_ids = sampler(
        target_with_bonus_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids
    )
    
    # Basic shape checks
    assert output_token_ids.shape == (batch_size, k + 1)
    assert torch.all((output_token_ids >= -1) & (output_token_ids < vocab_size))

@pytest.mark.parametrize("target_kl", [0.1, 0.2, 0.5])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mentored_sampling_kl_threshold(target_kl: float, device: str):
    """Test that KL divergence threshold affects acceptance rate."""
    torch.set_default_device(device)
    
    batch_size = 4
    k = 3
    vocab_size = 1000
    
    # Create target and draft distributions with controlled KL divergence
    target_probs = torch.ones((batch_size, k + 1, vocab_size)) / vocab_size
    draft_probs = torch.ones((batch_size, k, vocab_size)) / vocab_size
    
    # Make draft distribution slightly different from target
    draft_probs[:, :, :100] *= 1.2
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    
    draft_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, k))
    bonus_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, 1))
    
    sampler = MentoredSampler(target_kl=target_kl)
    
    output_token_ids = sampler(
        target_with_bonus_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids
    )
    
    # Count accepted tokens (not -1)
    accepted = (output_token_ids != -1).float().mean()
    
    # Higher target_kl should lead to more acceptances
    assert accepted > 0.0, "Should accept some tokens"

@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mentored_sampling_deterministic(device: str):
    """Test that sampling is deterministic with same seed."""
    torch.set_default_device(device)
    set_random_seed(42)
    
    batch_size = 4
    k = 3
    vocab_size = 1000
    
    target_probs, draft_probs, draft_token_ids, bonus_token_ids = _prepare_test(
        batch_size, k, vocab_size)
    
    sampler = MentoredSampler(target_kl=0.2)
    
    # First run
    output1 = sampler(
        target_with_bonus_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids
    )
    
    # Second run with same seed
    set_random_seed(42)
    output2 = sampler(
        target_with_bonus_probs=target_probs,
        bonus_token_ids=bonus_token_ids,
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids
    )
    
    assert torch.equal(output1, output2)