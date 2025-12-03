"""
Configuration for Teacher-Guided GRPO Pipeline
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class TeacherConfig:
    """Configuration for teacher model (vLLM)"""
    api_url: str = "http://localhost:8129/v1/chat/completions"
    model_name: str = "Qwen/Qwen3-32B-FP8"
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.5
    max_tokens: int = 2048
    enable_thinking: bool = True  # Critical for <think> reasoning blocks
    num_samples_per_problem: int = 8
    max_concurrent_requests: int = 32
    timeout: int = 60
    max_retries: int = 3


@dataclass
class StudentConfig:
    """Configuration for student model"""
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA alpha
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False


@dataclass
class GRPOConfig:
    """GRPO algorithm configuration"""
    num_ppo_epochs: int = 4
    num_mini_batches: int = 4
    kl_coef: float = 0.05
    clip_range: float = 0.2
    gamma: float = 1.0  # No discounting for code generation
    lam: float = 0.95
    value_clip_range: float = 0.2
    max_grad_norm: float = 1.0
    whiten_rewards: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 1024
    do_sample: bool = True
    num_samples_per_prompt: int = 4


@dataclass
class DualSourceConfig:
    """Dual-Source composite objective configuration"""
    # Loss weights
    grpo_weight: float = 1.0
    teacher_sft_weight: float = 0.3
    self_sft_weight: float = 0.2

    # Teacher-SFT parameters
    use_teacher_thinking: bool = True  # Include <think> blocks in SFT
    teacher_sft_max_length: int = 2048

    # Self-SFT parameters
    self_sft_top_k: int = 2  # Use top-k successful samples
    self_sft_min_reward: float = 0.8  # Minimum reward threshold
    self_sft_max_length: int = 1024


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "mbpp"
    train_split_ratio: float = 0.8
    validation_split_ratio: float = 0.2
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    seed: int = 42
    cache_dir: str = "./cache"
    synthetic_data_path: str = "./data/synthetic_traces.jsonl"


@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # Optimizer settings
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Logging
    logging_dir: str = "./logs"
    report_to: str = "wandb"
    run_name: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    k_values: list = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])
    num_samples_per_task: int = 100
    timeout: int = 5  # seconds per test case
    max_workers: int = 16
    temperature: float = 0.8
    top_p: float = 0.95


@dataclass
class MasterConfig:
    """Master configuration combining all configs"""
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dual_source: DualSourceConfig = field(default_factory=DualSourceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """Set torch dtype after initialization"""
        if self.student.torch_dtype == "bfloat16":
            self.student.torch_dtype = torch.bfloat16
        elif self.student.torch_dtype == "float16":
            self.student.torch_dtype = torch.float16
        else:
            self.student.torch_dtype = torch.float32


def get_default_config() -> MasterConfig:
    """Get default configuration"""
    return MasterConfig()
