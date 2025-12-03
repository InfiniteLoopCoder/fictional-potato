# Quick Start Guide

This guide will help you get started with the Teacher-Guided GRPO pipeline in 5 minutes.

## Prerequisites

1. **GPU**: NVIDIA GPU with at least 24GB VRAM (for student training)
2. **Python**: Python 3.8+
3. **Teacher Model**: Running vLLM server with Qwen3-32B-FP8

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x, CUDA: True
```

### 2. Start Teacher Model (vLLM)

**Option A: Using vLLM directly**

```bash
# Install vLLM if not already installed
pip install vllm

# Start server
vllm serve Qwen/Qwen3-32B-FP8 \
    --host 0.0.0.0 \
    --port 8129 \
    --dtype float16 \
    --tensor-parallel-size 1
```

**Option B: Using Docker**

```bash
docker run --gpus all \
    -p 8129:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-32B-FP8 \
    --dtype float16
```

### 3. Test Teacher Connection

```bash
python synthesis/teacher_query.py
```

Expected output:
```
Testing connection to teacher model at http://localhost:8129/v1/chat/completions...
✓ Connection successful!

Response preview:
Thinking: Let me break down this problem step by step...
Code: def add_numbers(a, b):
    return a + b
```

If this fails, check:
- Is vLLM server running? (`curl http://localhost:8129/health`)
- Is the port correct? (default: 8129)
- Is `enable_thinking` supported by your model?

### 4. Run the Pipeline

**Option A: Run all stages at once (recommended for first run)**

```bash
python main.py --stage all
```

This will:
1. Download MBPP dataset (~5 minutes)
2. Generate teacher traces (~30-60 minutes depending on dataset size)
3. Train student model (~2-4 hours depending on GPU)
4. Evaluate with Pass@k (~15-30 minutes)

**Option B: Run stages individually**

```bash
# Stage 1: Prepare data
python main.py --stage data

# Stage 2: Generate teacher traces (requires vLLM running)
python main.py --stage synthesis

# Stage 3: Train student model
python main.py --stage train

# Stage 4: Evaluate
python main.py --stage eval
```

**Option C: Use the shell script**

```bash
# Make executable
chmod +x run_pipeline.sh

# Run full pipeline with logging
./run_pipeline.sh all

# Or run specific stages
./run_pipeline.sh data
./run_pipeline.sh synthesis
./run_pipeline.sh train
./run_pipeline.sh eval
```

### 5. Monitor Training

**Terminal Output:**
```
==================================================
Training Configuration:
==================================================
Total epochs: 3
Steps per epoch: 125
Total training steps: 375
Batch size: 2
Gradient accumulation steps: 8
Training samples: 500
Teacher samples: 4000
==================================================

Epoch 1/3
Training: 100%|████████████| 125/125 [45:23<00:00, 21.79s/it]
```

**Weights & Biases (optional):**

To enable W&B logging:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Set in config.py: `training.report_to = "wandb"`

Then monitor at: https://wandb.ai/your-username/teacher-grpo

### 6. Check Results

After training completes:

```bash
# View evaluation results
cat outputs/evaluation_results.json

# Example output:
{
  "pass_at_k": {
    "1": 0.4523,
    "5": 0.6841,
    "10": 0.7592,
    "25": 0.8347
  },
  "num_tasks": 164,
  "num_samples_per_task": 100
}
```

**Interpret results:**
- **pass@1**: Single sample success rate (45.23%)
- **pass@5**: Success rate with 5 samples (68.41%)
- **pass@10**: Success rate with 10 samples (75.92%)

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution 1: Reduce batch size**

Edit `config.py`:
```python
training.per_device_train_batch_size = 1
training.gradient_accumulation_steps = 16
```

**Solution 2: Enable quantization**

Edit `config.py`:
```python
student.load_in_8bit = True
```

**Solution 3: Reduce LoRA rank**

Edit `config.py`:
```python
lora.r = 32
lora.lora_alpha = 64
```

### Issue 2: Teacher Connection Failed

**Check vLLM status:**
```bash
curl http://localhost:8129/health
curl http://localhost:8129/v1/models
```

**Update API URL:**

Edit `config.py`:
```python
teacher.api_url = "http://YOUR_SERVER:PORT/v1/chat/completions"
```

### Issue 3: Slow Synthesis

**Solution: Increase concurrency**

Edit `config.py`:
```python
teacher.max_concurrent_requests = 64  # Increase from 32
```

**Solution: Reduce samples per problem**

Edit `config.py`:
```python
teacher.num_samples_per_problem = 4  # Decrease from 8
```

### Issue 4: Low Pass@k Scores

**Try these adjustments:**

1. **Increase training epochs:**
```python
training.num_train_epochs = 5
```

2. **Increase LoRA rank:**
```python
lora.r = 128
lora.lora_alpha = 256
```

3. **Increase teacher-SFT weight:**
```python
dual_source.teacher_sft_weight = 0.5
```

4. **Generate more teacher samples:**
```python
teacher.num_samples_per_problem = 16
```

## Quick Configuration Examples

### Example 1: Fast Development Run

For quick testing with minimal resources:

```python
# config.py overrides
data.max_train_samples = 50
data.max_val_samples = 20
teacher.num_samples_per_problem = 2
training.num_train_epochs = 1
grpo.num_samples_per_prompt = 2
```

### Example 2: High-Quality Production Run

For best results with more resources:

```python
# config.py overrides
teacher.num_samples_per_problem = 16
teacher.max_concurrent_requests = 64
lora.r = 128
lora.lora_alpha = 256
training.num_train_epochs = 5
dual_source.teacher_sft_weight = 0.5
grpo.num_samples_per_prompt = 8
```

### Example 3: Memory-Constrained Setup

For training on GPUs with limited VRAM:

```python
# config.py overrides
student.load_in_8bit = True
training.per_device_train_batch_size = 1
training.gradient_accumulation_steps = 16
training.gradient_checkpointing = True
lora.r = 32
```

## Next Steps

1. **Customize Configuration**: Edit `config.py` to adjust hyperparameters
2. **Monitor Training**: Set up W&B for detailed metrics
3. **Experiment**: Try different loss weights and LoRA configurations
4. **Evaluate**: Test on your own code generation tasks
5. **Deploy**: Use the trained model for inference

## File Locations

After running the pipeline, you'll find:

```
grpo/
├── data/
│   ├── mbpp_train.jsonl          # Training data
│   ├── mbpp_validation.jsonl     # Validation data
│   ├── mbpp_test.jsonl           # Test data
│   └── synthetic_traces.jsonl    # Teacher-generated traces
│
├── outputs/
│   ├── checkpoint-500/           # Checkpoints
│   ├── checkpoint-1000/
│   ├── final_model/              # ← Use this for inference
│   └── evaluation_results.json   # Pass@k scores
│
└── logs/
    ├── stage1_data.log
    ├── stage2_synthesis.log
    ├── stage3_training.log
    └── stage4_evaluation.log
```

## Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")

# Generate code
prompt = "Write a function to find the nth Fibonacci number"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

outputs = model.generate(inputs, max_new_tokens=512, temperature=0.7)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

## Support

- **Issues**: Open an issue on GitHub
- **Documentation**: See README.md for detailed information
- **Configuration**: See config.py for all available options

## Minimal Working Example

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start vLLM (in another terminal)
vllm serve Qwen/Qwen3-32B-FP8 --port 8129

# 3. Test connection
python synthesis/teacher_query.py

# 4. Run pipeline
python main.py --stage all

# 5. Check results
cat outputs/evaluation_results.json
```

That's it! You now have a trained code generation model using Teacher-Guided GRPO.
