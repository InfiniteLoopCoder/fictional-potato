"""
Dual-Source composite objective: GRPO + Teacher-SFT + Self-SFT losses

IMPORTANT: SFT losses require properly constructed labels where:
- Prompt tokens have label = -100 (ignored in loss)
- Response tokens have label = token_id (used in loss)

This ensures we only compute loss on the assistant's response, not the user's prompt.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sft_data import prepare_teacher_sft_batch, prepare_self_sft_batch


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute supervised fine-tuning (SFT) loss

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        SFT loss scalar
    """
    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten tensors
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute loss
    loss = loss_fct(shift_logits, shift_labels)

    # Apply attention mask if provided
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous()
        loss = loss.view(shift_mask.shape)
        loss = (loss * shift_mask).sum() / shift_mask.sum()
    else:
        loss = loss.mean()

    return loss


def compute_teacher_sft_loss(
    model,
    teacher_examples: List[Dict],
    tokenizer,
    include_thinking: bool = True,
    max_length: int = 2048,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute Teacher-SFT loss from synthetic reasoning traces

    This properly constructs labels where:
    - User prompt tokens are masked with -100 (no loss)
    - Assistant response tokens use actual token IDs (compute loss)

    Args:
        model: Student model
        teacher_examples: List of teacher trace dicts with 'prompt', 'thinking', 'code'
        tokenizer: Tokenizer
        include_thinking: Whether to include thinking blocks in SFT target
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        Teacher-SFT loss
    """
    # Prepare batch with proper labels
    batch = prepare_teacher_sft_batch(
        teacher_examples=teacher_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        include_thinking=include_thinking,
    )

    # Move to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Compute SFT loss (CrossEntropyLoss handles -100 labels automatically)
    loss = compute_sft_loss(
        logits=outputs.logits,
        labels=labels,
        attention_mask=attention_mask,
    )

    return loss


def compute_self_sft_loss(
    model,
    successful_samples: List[Dict],
    tokenizer,
    prompts: List[str],
    top_k: int = 2,
    min_reward: float = 0.8,
    max_length: int = 1024,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """
    Compute Self-SFT loss from best successful samples

    This properly constructs labels where:
    - User prompt tokens are masked with -100 (no loss)
    - Generated response tokens use actual token IDs (compute loss)

    IMPORTANT: We only want to compute loss on the model's RESPONSE,
    not on the original prompt!

    Args:
        model: Student model
        successful_samples: List of dicts with 'prompt', 'response', 'reward'
        tokenizer: Tokenizer
        prompts: Original prompts (for reference, can be empty)
        top_k: Number of top samples to use
        min_reward: Minimum reward threshold
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        Self-SFT loss or None if no successful samples
    """
    if not successful_samples:
        return None

    # Prepare batch with proper labels
    batch = prepare_self_sft_batch(
        successful_samples=successful_samples,
        tokenizer=tokenizer,
        prompts=prompts,
        top_k=top_k,
        min_reward=min_reward,
        max_length=max_length,
    )

    if batch is None:
        return None

    # Move to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Compute SFT loss (CrossEntropyLoss handles -100 labels automatically)
    loss = compute_sft_loss(
        logits=outputs.logits,
        labels=labels,
        attention_mask=attention_mask,
    )

    return loss


def compute_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    kl_coef: float = 0.05,
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss

    Args:
        log_probs: Current policy log probabilities
        ref_log_probs: Reference policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage values
        kl_coef: KL divergence coefficient
        clip_range: PPO clipping range

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Compute ratio for PPO
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range
    )
    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

    # KL divergence with reference model
    kl_div = (ref_log_probs - log_probs).mean()
    kl_penalty = kl_coef * kl_div

    # Total GRPO loss
    total_loss = policy_loss + kl_penalty

    # Compute metrics
    metrics = {
        "grpo/policy_loss": policy_loss.item(),
        "grpo/kl_div": kl_div.item(),
        "grpo/kl_penalty": kl_penalty.item(),
        "grpo/ratio_mean": ratio.mean().item(),
        "grpo/ratio_std": ratio.std().item(),
        "grpo/clip_fraction": ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean().item(),
    }

    return total_loss, metrics


def compute_dual_source_loss(
    grpo_loss: torch.Tensor,
    teacher_sft_loss: Optional[torch.Tensor],
    self_sft_loss: Optional[torch.Tensor],
    grpo_weight: float = 1.0,
    teacher_sft_weight: float = 0.3,
    self_sft_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Dual-Source composite objective

    Combines:
    1. GRPO loss (execution-feedback)
    2. Teacher-SFT loss (static synthetic traces)
    3. Self-SFT loss (dynamic best samples)

    Args:
        grpo_loss: GRPO loss
        teacher_sft_loss: Teacher-SFT loss (can be None)
        self_sft_loss: Self-SFT loss (can be None)
        grpo_weight: Weight for GRPO loss
        teacher_sft_weight: Weight for Teacher-SFT loss
        self_sft_weight: Weight for Self-SFT loss

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Start with GRPO loss
    total_loss = grpo_weight * grpo_loss

    metrics = {
        "loss/grpo": grpo_loss.item(),
        "loss/grpo_weighted": (grpo_weight * grpo_loss).item(),
    }

    # Add Teacher-SFT loss if available
    if teacher_sft_loss is not None:
        weighted_teacher_sft = teacher_sft_weight * teacher_sft_loss
        total_loss = total_loss + weighted_teacher_sft

        metrics["loss/teacher_sft"] = teacher_sft_loss.item()
        metrics["loss/teacher_sft_weighted"] = weighted_teacher_sft.item()

    # Add Self-SFT loss if available
    if self_sft_loss is not None:
        weighted_self_sft = self_sft_weight * self_sft_loss
        total_loss = total_loss + weighted_self_sft

        metrics["loss/self_sft"] = self_sft_loss.item()
        metrics["loss/self_sft_weighted"] = weighted_self_sft.item()

    metrics["loss/total"] = total_loss.item()

    return total_loss, metrics


def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """
    Whiten values (normalize to mean 0, std 1)

    Args:
        values: Input tensor
        shift_mean: Whether to shift mean to 0

    Returns:
        Whitened tensor
    """
    mean = values.mean()
    std = values.std()

    if shift_mean:
        values = (values - mean) / (std + 1e-8)
    else:
        values = values / (std + 1e-8)

    return values


def compute_advantages(
    rewards: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
    whiten_rewards: bool = True,
) -> torch.Tensor:
    """
    Compute advantage estimates (GAE if values provided, else direct rewards)

    Args:
        rewards: Reward values
        values: Value estimates (optional)
        gamma: Discount factor
        lam: GAE lambda
        whiten_rewards: Whether to whiten advantages

    Returns:
        Advantage tensor
    """
    if values is None:
        # Use rewards directly as advantages
        advantages = rewards
    else:
        # Compute GAE
        advantages = rewards - values

    # Whiten advantages
    if whiten_rewards:
        advantages = whiten(advantages)

    return advantages
