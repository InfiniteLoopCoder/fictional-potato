"""
Training utilities for GRPO
"""
import torch
import random
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RolloutBatch:
    """Batch of rollout data for GRPO training"""
    queries: torch.Tensor  # Input prompts
    responses: torch.Tensor  # Generated responses
    log_probs: torch.Tensor  # Log probabilities
    rewards: torch.Tensor  # Rewards from execution
    advantages: torch.Tensor  # Computed advantages
    input_ids: torch.Tensor  # Full input_ids (query + response)
    attention_mask: torch.Tensor  # Attention mask


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_model_weights(model, freeze_layers: Optional[List[str]] = None):
    """
    Freeze model weights

    Args:
        model: Model to freeze
        freeze_layers: List of layer names to freeze (None for all)
    """
    if freeze_layers is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False


def get_trainable_parameters(model) -> int:
    """Get number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_parameters(model):
    """Print trainable parameters statistics"""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable %: {100 * trainable_params / all_params:.2f}%"
    )


def prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool = True):
    """
    Prepare model for k-bit training

    Args:
        model: Model to prepare
        use_gradient_checkpointing: Whether to enable gradient checkpointing
    """
    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Enable input gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def gather_rollout_stats(rewards: List[float], episode_lengths: List[int]) -> Dict[str, float]:
    """
    Gather statistics from rollouts

    Args:
        rewards: List of rewards
        episode_lengths: List of episode lengths

    Returns:
        Dictionary of statistics
    """
    return {
        "rollout/reward_mean": np.mean(rewards),
        "rollout/reward_std": np.std(rewards),
        "rollout/reward_min": np.min(rewards),
        "rollout/reward_max": np.max(rewards),
        "rollout/episode_length_mean": np.mean(episode_lengths),
        "rollout/episode_length_std": np.std(episode_lengths),
    }


def format_prompt_for_chat(tokenizer, task_prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Format prompt for chat model

    Args:
        tokenizer: Tokenizer
        task_prompt: Task prompt
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": task_prompt})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return formatted


def compute_response_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    """
    Compute log probabilities for generated responses

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        query_length: Length of query (prompt)

    Returns:
        Log probabilities [batch_size]
    """
    # Get response portion
    response_ids = input_ids[:, query_length:]
    response_logits = logits[:, query_length - 1:-1, :]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)

    # Gather log probs for actual tokens
    log_probs = torch.gather(
        log_probs,
        2,
        response_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probs over sequence
    return log_probs.sum(dim=-1)


def create_reference_model(model, device: str = "cuda"):
    """
    Create reference model for KL penalty

    Args:
        model: Base model
        device: Device to use

    Returns:
        Reference model (frozen copy)
    """
    import copy

    ref_model = copy.deepcopy(model)

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    ref_model.eval()

    return ref_model


def save_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    epoch: int,
    step: int,
    optimizer=None,
    scheduler=None,
):
    """
    Save training checkpoint

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        epoch: Current epoch
        step: Current step
        optimizer: Optimizer state (optional)
        scheduler: Scheduler state (optional)
    """
    from pathlib import Path

    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training state
    if optimizer is not None or scheduler is not None:
        state = {
            "epoch": epoch,
            "step": step,
        }

        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        torch.save(state, checkpoint_dir / "training_state.pt")

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(
    model,
    checkpoint_dir: str,
    optimizer=None,
    scheduler=None,
):
    """
    Load training checkpoint

    Args:
        model: Model to load into
        checkpoint_dir: Checkpoint directory
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Dictionary with epoch and step
    """
    from pathlib import Path

    checkpoint_path = Path(checkpoint_dir) / "training_state.pt"

    if not checkpoint_path.exists():
        return {"epoch": 0, "step": 0}

    state = torch.load(checkpoint_path)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return {
        "epoch": state.get("epoch", 0),
        "step": state.get("step", 0),
    }
