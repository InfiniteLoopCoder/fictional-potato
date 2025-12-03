# Teacher-Guided GRPO for Code Generation

A complete implementation of Teacher-Guided Group Relative Policy Optimization (GRPO) for training code generation models using the MBPP dataset.

## Overview

This pipeline implements a state-of-the-art approach to training code generation models by combining:

1. **Offline Teacher Synthesis**: Generate high-quality reasoning traces using a larger teacher model (Qwen3-32B-FP8)
2. **GRPO with LoRA**: Efficient fine-tuning using Low-Rank Adaptation to prevent catastrophic forgetting
3. **Dual-Source Objective**: Composite loss combining:
   - GRPO execution-feedback loss
   - Teacher-SFT loss (from synthetic reasoning traces)
   - Self-SFT loss (from best successful samples)
4. **Pass@k Evaluation**: Rigorous evaluation on held-out test set

## Architecture

```
Teacher Model (Qwen3-32B-FP8)
    ↓ [generates reasoning traces with <think> blocks]
Synthetic Training Data
    ↓
Student Model (Qwen2.5-Coder-1.5B-Instruct) + LoRA
    ↓ [trained with Dual-Source objective]
    ├── GRPO Loss (execution rewards)
    ├── Teacher-SFT Loss (reasoning traces)
    └── Self-SFT Loss (best samples)
    ↓
Optimized Code Generation Model
```

## Features

- ✅ **High-Concurrency Teacher Queries**: Async API calls with configurable concurrency limits
- ✅ **Thinking Mode**: Extracts `<think>` reasoning blocks from teacher responses
- ✅ **LoRA Adapters**: Memory-efficient training with configurable rank and alpha
- ✅ **Custom GRPO Trainer**: Group-based policy optimization with KL penalties
- ✅ **Dual-Source Learning**: Combines execution feedback with knowledge distillation
- ✅ **Secure Code Execution**: Sandboxed test execution with timeouts
- ✅ **Pass@k Metrics**: Standard code generation benchmarking
- ✅ **Full Parameter Support**: All hyperparameters configurable via config file

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import transformers; import peft; print('✓ Installation successful')"
```

## Quick Start

### 1. Start Teacher Model (vLLM)

First, ensure your teacher model is running at `http://localhost:8129`:

```bash
# Example vLLM command (adjust paths as needed)
vllm serve Qwen/Qwen3-32B-FP8 \
    --host 0.0.0.0 \
    --port 8129 \
    --dtype float16
```

### 2. Test Teacher Connection

```bash
python synthesis/teacher_query.py
```

Expected output:
```
Testing connection to teacher model at http://localhost:8129/v1/chat/completions...
✓ Connection successful!

Response preview:
Thinking: Let me break down this problem...
Code: def add_numbers(a, b):...
```

### 3. Run Full Pipeline

```bash
# Run all stages: data → synthesis → training → evaluation
python main.py --stage all

# Or run stages individually:
python main.py --stage data        # Download and split MBPP
python main.py --stage synthesis   # Generate teacher traces
python main.py --stage train       # Train with GRPO
python main.py --stage eval        # Evaluate with Pass@k
```

## Configuration

### Default Configuration

All hyperparameters are defined in `config.py`. Key settings:

```python
# Teacher Model
teacher.model_name = "Qwen/Qwen3-32B-FP8"
teacher.temperature = 0.7
teacher.top_p = 0.8
teacher.presence_penalty = 1.5
teacher.enable_thinking = True  # Critical for <think> blocks

# Student Model
student.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# LoRA Configuration
lora.r = 64
lora.lora_alpha = 128
lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ...]

# GRPO
grpo.kl_coef = 0.05
grpo.clip_range = 0.2
grpo.num_samples_per_prompt = 4

# Dual-Source Weights
dual_source.grpo_weight = 1.0
dual_source.teacher_sft_weight = 0.3
dual_source.self_sft_weight = 0.2
```

### Custom Configuration

Create a custom JSON config:

```json
{
  "teacher": {
    "temperature": 0.8,
    "num_samples_per_problem": 16
  },
  "lora": {
    "r": 128,
    "lora_alpha": 256
  },
  "dual_source": {
    "teacher_sft_weight": 0.5
  }
}
```

Run with custom config:

```bash
python main.py --config my_config.json
```

## Pipeline Stages

### Stage 1: Data Preparation

Downloads MBPP dataset and splits into train/validation/test:

```bash
python main.py --stage data
```

Output:
- `data/mbpp_train.jsonl`: Training set (80%)
- `data/mbpp_validation.jsonl`: Validation set (20%)
- `data/mbpp_test.jsonl`: Held-out test set

### Stage 2: Teacher Synthesis

Queries teacher model to generate reasoning traces:

```bash
python main.py --stage synthesis
```

**Critical API Payload Structure:**

```python
{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [...],
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "max_tokens": 2048,
    "chat_template_kwargs": {
        "enable_thinking": true  # REQUIRED for <think> blocks
    }
}
```

Output:
- `data/synthetic_traces.jsonl`: 8 samples per problem with reasoning

### Stage 3: GRPO Training

Trains student model with LoRA and Dual-Source objective:

```bash
python main.py --stage train
```

Training loop:
1. Generate responses from current policy
2. Execute code against test cases → rewards
3. Compute GRPO loss with KL penalty
4. Sample teacher traces → Teacher-SFT loss
5. Select best samples → Self-SFT loss
6. Optimize composite objective
7. Repeat

Output:
- `outputs/checkpoint-{step}/`: Periodic checkpoints
- `outputs/final_model/`: Final trained model

### Stage 4: Evaluation

Evaluates model using Pass@k metrics:

```bash
python main.py --stage eval

# Or evaluate specific checkpoint:
python main.py --eval_only --model_path outputs/checkpoint-1000
```

Output:
```
==================================================
Pass@k Results:
==================================================
pass@  1: 0.4523 (45.23%)
pass@  5: 0.6841 (68.41%)
pass@ 10: 0.7592 (75.92%)
pass@ 25: 0.8347 (83.47%)
==================================================
```

## Project Structure

```
grpo/
├── config.py                    # Central configuration
├── main.py                      # Main pipeline script
├── requirements.txt             # Dependencies
│
├── data/                        # Dataset handling
│   ├── download_mbpp.py        # MBPP download & split
│   ├── mbpp_train.jsonl        # Training data
│   ├── mbpp_validation.jsonl   # Validation data
│   └── mbpp_test.jsonl         # Test data
│
├── synthesis/                   # Teacher synthesis
│   ├── teacher_query.py        # vLLM API client
│   ├── generate_traces.py      # Trace generation
│   └── synthetic_traces.jsonl  # Generated traces
│
├── training/                    # GRPO training
│   ├── grpo_trainer.py         # Custom GRPO trainer
│   ├── losses.py               # Dual-Source losses
│   └── utils.py                # Training utilities
│
├── evaluation/                  # Evaluation
│   ├── code_executor.py        # Sandboxed execution
│   └── pass_at_k.py            # Pass@k metrics
│
└── outputs/                     # Training outputs
    ├── checkpoint-*/            # Checkpoints
    ├── final_model/             # Final model
    └── evaluation_results.json  # Eval results
```

## Key Implementation Details

### 1. Teacher Query with Thinking Mode

The pipeline uses a specific API payload structure to enable reasoning:

```python
payload = {
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": messages,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {
        "enable_thinking": true  # Enables <think> blocks
    }
}
```

Response format:
```
<think>
First, I need to understand the problem...
The approach is to use dynamic programming...
</think>

def solve_problem(n):
    # Implementation
    ...
```

### 2. LoRA Configuration

LoRA adapters are applied to all attention and MLP layers:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # MLP
]
```

Benefits:
- Reduces trainable parameters by ~99%
- Prevents catastrophic forgetting
- Enables efficient training on consumer GPUs

### 3. Dual-Source Objective

The composite loss combines three components:

```python
L_total = α·L_GRPO + β·L_TeacherSFT + γ·L_SelfSFT

where:
  L_GRPO = PPO loss + KL penalty (execution feedback)
  L_TeacherSFT = Cross-entropy on teacher traces (static)
  L_SelfSFT = Cross-entropy on best samples (dynamic)
```

Default weights: α=1.0, β=0.3, γ=0.2

### 4. Reward Computation

Code is executed in isolated processes with timeouts:

```python
reward = compute_reward(
    code=generated_code,
    test_cases=["assert func(1,2)==3", ...],
    timeout=5,
    reward_type="binary"  # 1.0 if all pass, 0.0 otherwise
)
```

## Advanced Usage

### Custom Teacher Model

To use a different teacher model:

1. Update `config.py`:
```python
teacher.api_url = "http://your-server:port/v1/chat/completions"
teacher.model_name = "your/model-name"
```

2. Ensure API supports `chat_template_kwargs.enable_thinking`

### Multi-GPU Training

The trainer uses `device_map="auto"` for automatic multi-GPU:

```python
student.device_map = "auto"  # Automatically distributes across GPUs
```

For distributed training, use `accelerate`:

```bash
accelerate config  # Configure multi-GPU setup
accelerate launch main.py --stage train
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **LoRA Rank** (`lora.r`): Higher = more capacity, slower training
   - Try: 16, 32, 64, 128

2. **KL Coefficient** (`grpo.kl_coef`): Controls exploration
   - Try: 0.01, 0.05, 0.1

3. **Loss Weights** (`dual_source.*_weight`): Balance objectives
   - Increase teacher_sft_weight for more imitation
   - Increase grpo_weight for more execution focus

4. **Temperature** (`teacher.temperature`, `grpo.temperature`):
   - Higher = more diverse generations
   - Try: 0.6-1.0

### Monitoring Training

The pipeline supports Weights & Biases logging:

```python
training.report_to = "wandb"
training.run_name = "my-experiment"
```

Logged metrics:
- `loss/total`: Total composite loss
- `loss/grpo`: GRPO component
- `loss/teacher_sft`: Teacher-SFT component
- `loss/self_sft`: Self-SFT component
- `grpo/kl_div`: KL divergence from reference
- `rollout/reward_mean`: Average execution reward

## Troubleshooting

### Teacher Connection Issues

```bash
# Test connection
python synthesis/teacher_query.py

# Common issues:
# 1. vLLM not running → start vLLM server
# 2. Wrong port → check config.teacher.api_url
# 3. Missing enable_thinking → verify chat_template_kwargs
```

### CUDA Out of Memory

Reduce memory usage:

```python
# Smaller batch size
training.per_device_train_batch_size = 1
training.gradient_accumulation_steps = 16

# Enable gradient checkpointing
training.gradient_checkpointing = True

# Use 8-bit or 4-bit quantization
student.load_in_8bit = True
```

### Low Pass@k Scores

Try:
1. Increase teacher samples: `teacher.num_samples_per_problem = 16`
2. Increase LoRA rank: `lora.r = 128`
3. Increase teacher-SFT weight: `dual_source.teacher_sft_weight = 0.5`
4. Train longer: `training.num_train_epochs = 5`

## Citation

If you use this code, please cite:

```bibtex
@software{teacher_grpo_2024,
  title={Teacher-Guided GRPO for Code Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/grpo}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- MBPP dataset: [Mostly Basic Python Problems](https://github.com/google-research/google-research/tree/master/mbpp)
- vLLM: [Fast LLM Inference](https://github.com/vllm-project/vllm)
- Qwen Models: [Alibaba Cloud](https://github.com/QwenLM/Qwen)
- PEFT/LoRA: [HuggingFace](https://github.com/huggingface/peft)
