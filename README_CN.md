# Teacher-Guided GRPO - 代码生成训练系统

基于MBPP和APPS数据集的教师引导式强化学习代码生成系统。

---

## 核心特性

✅ **统一提示词格式** - 所有阶段使用相同格式(thinking + markdown代码)
✅ **双源学习目标** - GRPO + Teacher-SFT + Self-SFT
✅ **LoRA高效训练** - 参数减少99%,防止灾难性遗忘
✅ **多数据集支持** - MBPP + APPS混合训练
✅ **完整测试脚本** - 验证所有假设和提取流程

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 测试环境

```bash
# 运行完整测试(必须!)
python test_all.py
```

**测试内容**:
- Student模型输出格式
- Teacher模型(vLLM)提取
- Thinking和代码提取
- SFT标签构建
- 代码执行流程
- 所有提取方法一致性

### 3. 启动Teacher模型(vLLM)

```bash
vllm serve Qwen/Qwen3-32B-FP8 --port 8129 --dtype float16
```

### 4. 运行训练流程

```bash
# 阶段1: 准备数据(MBPP + APPS, 80/20划分)
python phase1_prepare_data.py

# 阶段2: 生成Teacher推理轨迹
python phase2_generate_teacher_traces.py

# 阶段3: GRPO训练(thinking + code)
python phase3_train_grpo.py

# 阶段4: 评估(Pass@k)
python phase4_evaluate.py
```

---

## 系统架构

### 统一提示词(所有阶段相同)

```
解决以下问题:

{problem}

请在<think>标签中逐步思考解决方案,然后用markdown格式提供Python代码。

格式:
<think>
你的逐步推理...
</think>

```python
# 你的代码
```
```

**为什么这样设计?**

1. **Thinking块** (`<think>...</think>`)
   - 鼓励推理后再编码
   - 可被提取用于Teacher-SFT
   - RL同时优化thinking和code

2. **Markdown代码块** (` ```python ... ``` `)
   - 标准格式,易于解析
   - 分离代码和其他文本
   - 鲁棒提取

3. **一致性**
   - 相同格式 → 模型学习统一模式
   - Teacher/RL/Eval使用相同提示

### 数据流程

```
MBPP数据集 (500个问题)
    +
APPS数据集 (5000个问题)
    ↓ [80/20划分]
训练集 + 验证集
    ↓
Teacher模型(Qwen3-32B-FP8)
    ↓ [生成thinking + code]
合成推理轨迹 (每题8个样本)
    ↓
Student模型(Qwen2.5-Coder-1.5B) + LoRA
    ↓ [Dual-Source训练]
    ├─ GRPO Loss (执行奖励)
    ├─ Teacher-SFT Loss (推理轨迹)
    └─ Self-SFT Loss (最佳样本)
    ↓
优化的代码生成模型
    ↓
Pass@k评估
```

---

## 核心概念

### 1. 统一提示词

**问题**: 之前Teacher/RL/Eval使用不同提示 → 输出不一致

**解决**: 所有阶段使用 `get_unified_prompt()`

```python
from utils.prompts import get_unified_prompt, SYSTEM_PROMPT_UNIFIED

# 在所有地方使用
prompt = get_unified_prompt(task_description)
system = SYSTEM_PROMPT_UNIFIED
```

### 2. RL优化Thinking + Code

**关键**: RL不仅优化代码,还优化推理过程

```python
# GRPO训练循环
response = model.generate(unified_prompt)
# 输出: <think>推理...</think>\n\n```python\n代码...```

code = extract_code(response)  # 提取代码部分
reward = execute(code, tests)  # 执行获得奖励

log_probs = model.log_probs(full_response)  # 完整序列的log概率
loss = grpo_loss(log_probs, reward)  # 更新thinking + code的权重
```

### 3. SFT标签构建(重要!)

**错误做法**:
```python
labels = input_ids.clone()  # 包含prompt和response!
# 问题: 模型会学习用户的提问,数据泄露!
```

**正确做法**:
```python
# Prompt部分: label = -100 (loss中忽略)
# Response部分: label = token_id (计算loss)

for i, token_id in enumerate(input_ids):
    if i < prompt_length:
        labels.append(-100)      # 忽略prompt
    else:
        labels.append(token_id)  # 学习response
```

**可视化**:
```
序列: <user>写一个函数<assistant><think>推理...</think>\ndef add(a,b)...

Labels: [-100][-100][-100]...[-100][token_ids for response...]
         ^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^
         Prompt (忽略)              Response (学习)
```

### 4. 代码提取流程

**支持多种格式**:

```python
# 输入: 模型完整输出
response = """
<think>
检查质数:
1. n <= 1 不是质数
2. 检查到sqrt(n)
</think>

```python
def is_prime(n):
    if n <= 1:
        return False
    ...
```
"""

# 提取
from utils.thinking_extraction import extract_thinking_and_code_unified

parsed = extract_thinking_and_code_unified(response)
# {
#   'thinking': "检查质数:\n1. n <= 1...",
#   'code': "def is_prime(n):\n    if n <= 1:...",
# }

# 执行
from evaluation.code_executor import extract_function_from_code
clean_code = extract_function_from_code(response)
# 返回: 纯净代码,去除thinking和markdown

reward = compute_reward(clean_code, test_cases)
```

---

## 配置说明

所有超参数在 `config.py`:

### Teacher配置

```python
teacher.api_url = "http://localhost:8129/v1/chat/completions"
teacher.model_name = "Qwen/Qwen3-32B-FP8"
teacher.temperature = 0.7
teacher.top_p = 0.8
teacher.presence_penalty = 1.5
teacher.enable_thinking = True  # 启用thinking模式
teacher.num_samples_per_problem = 8  # 每题生成8个样本
teacher.max_concurrent_requests = 32  # 并发请求数
```

### Student配置

```python
student.model_name = "Qwen/Qwen3-1.7B"
student.torch_dtype = torch.bfloat16
```

### LoRA配置

```python
lora.r = 64              # LoRA秩
lora.lora_alpha = 128    # LoRA alpha (通常为2×r)
lora.target_modules = [  # 要适配的层
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### GRPO配置

```python
grpo.kl_coef = 0.05           # KL散度系数
grpo.clip_range = 0.2         # PPO裁剪范围
grpo.num_samples_per_prompt = 4  # 每个prompt采样数
```

### Dual-Source权重

```python
dual_source.grpo_weight = 1.0         # GRPO loss权重
dual_source.teacher_sft_weight = 0.3  # Teacher-SFT权重
dual_source.self_sft_weight = 0.2     # Self-SFT权重
```

### 训练配置

```python
training.num_train_epochs = 3
training.per_device_train_batch_size = 2
training.gradient_accumulation_steps = 8
training.learning_rate = 5e-5
```

---

## 阶段详解

### 阶段1: 数据准备

**脚本**: `phase1_prepare_data.py`

**功能**:
- 下载MBPP和APPS数据集
- 各自80/20划分
- 合并: (80% MBPP + 80% APPS) → 训练集
- 合并: (20% MBPP + 20% APPS) → 验证集

**用法**:
```bash
# 包含所有APPS难度
python phase1_prepare_data.py

# 仅包含简单APPS题目
python phase1_prepare_data.py --apps_difficulty introductory

# 仅使用MBPP
python phase1_prepare_data.py --skip_apps
```

**输出**:
- `data/train.jsonl`: 训练集
- `data/val.jsonl`: 验证集

### 阶段2: 生成Teacher轨迹

**脚本**: `phase2_generate_teacher_traces.py`

**功能**:
- 使用统一提示词查询Teacher模型(vLLM)
- 提取thinking块和代码
- 验证Python语法
- 保存合成轨迹

**用法**:
```bash
# 测试连接
python phase2_generate_teacher_traces.py --test_connection

# 生成轨迹
python phase2_generate_teacher_traces.py

# 测试(仅5个任务)
python phase2_generate_teacher_traces.py --max_tasks 5 --num_samples 2
```

**输出**:
- `data/teacher_traces.jsonl`: Teacher推理轨迹(每题8个样本)

**统计示例**:
```
总响应数:        3200
包含thinking:    3150 (98.4%)
提取到代码:      3180 (99.4%)
有效Python:      3170 (99.1%)
解析失败:        20 (0.6%)
```

### 阶段3: GRPO训练

**脚本**: `phase3_train_grpo.py`

**功能**:
- 加载Student模型 + 应用LoRA
- 使用统一提示词生成(thinking + code)
- 执行代码获取奖励
- Dual-Source目标优化:
  - GRPO: 基于执行奖励
  - Teacher-SFT: 学习Teacher轨迹
  - Self-SFT: 学习自己的成功样本

**训练循环**:
```python
for task in train_data:
    # 1. 生成(thinking + code)
    response = model.generate(unified_prompt(task))

    # 2. 提取代码执行
    code = extract_code(response)
    reward = execute(code, task.tests)

    # 3. 计算GRPO loss
    log_probs = model.log_probs(full_response)  # 完整序列
    grpo_loss = ppo_loss(log_probs, reward) + kl_penalty

    # 4. Teacher-SFT loss
    teacher_batch = sample(teacher_traces)
    teacher_loss = sft_loss(model, teacher_batch)

    # 5. Self-SFT loss
    best_samples = top_k(successful_responses)
    self_loss = sft_loss(model, best_samples)

    # 6. 组合优化
    total_loss = 1.0*grpo_loss + 0.3*teacher_loss + 0.2*self_loss
    total_loss.backward()
```

**输出**:
- `outputs/checkpoint-*/`: 定期检查点
- `outputs/final_model/`: 最终LoRA模型

### 阶段4: 评估

**脚本**: `phase4_evaluate.py`

**功能**:
- 使用统一提示词生成代码
- 执行测试用例
- 计算Pass@k指标

**用法**:
```bash
# 评估最终模型
python phase4_evaluate.py

# 评估特定检查点
python phase4_evaluate.py --model_path outputs/checkpoint-1000

# 生成更多样本(更准确的估计)
python phase4_evaluate.py --num_samples 200
```

**输出**:
```
==================================================
Pass@k Results:
==================================================
pass@  1: 0.4523 (45.23%)
pass@  5: 0.6841 (68.41%)
pass@ 10: 0.7592 (75.92%)
==================================================
```

---

## 测试脚本

### test_all.py - 完整验证

**必须运行!** 在训练前验证所有假设:

```bash
python test_all.py
```

**测试项目**:

1. **Student模型输出**
   - 是否输出`<think>`块?
   - 是否使用markdown?
   - 提取是否正确?

2. **Teacher(vLLM)提取**
   - thinking提取正确?
   - 代码提取正确?
   - 支持多种格式?

3. **代码提取一致性**
   - 多种提取方法结果一致?
   - 提取的代码语法正确?

4. **SFT标签构建**
   - Prompt被mask(-100)?
   - Response有正确token_id?
   - 无数据泄露?

5. **代码执行**
   - 从完整响应提取代码?
   - 执行测试用例?
   - 计算奖励?

6. **Teacher API**
   - vLLM连接正常?(如果运行)
   - 返回thinking和代码?

7. **统一提示词**
   - 包含关键元素?
   - 系统提示词正确?

**输出示例**:
```
✓ PASS 模型加载成功
✓ PASS 输出包含 <think> 块
✓ PASS 成功提取thinking (120 chars)
✓ PASS 成功提取代码 (250 chars)
✓ PASS 代码语法有效
✓ PASS 所有提取方法一致
✓ PASS SFT批次准备成功
✓ PASS 标签包含提示和响应部分
✓ PASS 所有测试用例通过
✓ PASS 奖励计算正确

测试总结:
总测试数: 25
通过: 25
失败: 0
警告: 0

所有测试通过,可以开始训练!
```

---

## 项目结构

```
grpo/
├── 配置与脚本
│   ├── config.py                    # 所有超参数
│   ├── requirements.txt             # 依赖
│   ├── test_all.py                  # ⭐ 完整测试脚本
│   └── README_CN.md                 # ⭐ 本文档
│
├── 阶段脚本
│   ├── phase1_prepare_data.py       # 数据准备
│   ├── phase2_generate_teacher_traces.py  # Teacher轨迹生成
│   ├── phase3_train_grpo.py         # GRPO训练
│   └── phase4_evaluate.py           # Pass@k评估
│
├── utils/                           # 工具模块
│   ├── prompts.py                   # 统一提示词
│   ├── thinking_extraction.py       # Thinking提取
│   ├── code_parser.py               # 代码解析
│   ├── sft_data.py                  # SFT数据准备
│   └── dataset_loader.py            # MBPP+APPS加载
│
├── synthesis/                       # Teacher合成
│   ├── teacher_query.py             # vLLM API客户端
│   └── generate_traces.py           # 轨迹生成
│
├── training/                        # 训练模块
│   ├── grpo_trainer.py              # GRPO训练器
│   ├── losses.py                    # Dual-Source losses
│   └── utils.py                     # 训练工具
│
├── evaluation/                      # 评估模块
│   ├── code_executor.py             # 代码执行
│   └── pass_at_k.py                 # Pass@k指标
│
└── 数据输出
    ├── data/
    │   ├── train.jsonl              # 训练集
    │   ├── val.jsonl                # 验证集
    │   └── teacher_traces.jsonl     # Teacher轨迹
    └── outputs/
        ├── checkpoint-*/            # 检查点
        ├── final_model/             # 最终模型
        └── evaluation_results.json  # 评估结果
```

---

## 常见问题

### Q1: RL是否优化thinking和代码?

**A: 是的!**

- 使用统一提示词 → 生成thinking + code
- 代码部分执行 → 获得奖励
- log_probs在完整序列上 → RL同时优化thinking和code

### Q2: 代码是否使用markdown格式?

**A: 是的!**

- 提示词明确要求: "用markdown格式提供代码"
- 提取逻辑处理:
  - ` ```python ... ``` ` (首选)
  - ` ``` ... ``` ` (备选)
  - 纯文本 (备选)
- 所有阶段使用相同提取

### Q3: SFT如何构建标签?

**A: Prompt被mask,只学习Response**

```python
# 正确做法
labels[prompt_part] = -100        # 忽略prompt
labels[response_part] = token_id  # 学习response
```

### Q4: 如何测试所有假设?

**A: 运行test_all.py**

```bash
python test_all.py
```

验证:
- 模型输出格式
- 提取流程正确性
- SFT标签构建
- 代码执行
- Teacher API

### Q5: CUDA内存不足?

**A: 减少批次大小或启用量化**

```python
# config.py
training.per_device_train_batch_size = 1
training.gradient_accumulation_steps = 16
student.load_in_8bit = True
```

### Q6: Pass@k分数低?

**A: 尝试以下调整**

1. 增加训练轮数: `training.num_train_epochs = 5`
2. 增加Teacher样本: `teacher.num_samples_per_problem = 16`
3. 增加Teacher-SFT权重: `dual_source.teacher_sft_weight = 0.5`
4. 增加LoRA秩: `lora.r = 128`

---

## 使用流程总结

### 1. 测试环境(必须!)

```bash
python test_all.py
```

确保所有测试通过。

### 2. 启动vLLM

```bash
vllm serve Qwen/Qwen3-32B-FP8 --port 8129
```

### 3. 运行4个阶段

```bash
python phase1_prepare_data.py
python phase2_generate_teacher_traces.py
python phase3_train_grpo.py
python phase4_evaluate.py
```

### 4. 检查结果

```bash
cat outputs/evaluation_results.json
```

---

## 核心优势

✅ **统一格式** - 所有阶段相同提示词,模型行为一致
✅ **完整RL** - 优化thinking + code,不仅仅是代码
✅ **正确SFT** - 无prompt泄露,只学习响应
✅ **鲁棒提取** - 多种格式回退,处理各种输出
✅ **完整测试** - test_all.py验证所有假设
✅ **高效训练** - LoRA减少99%参数,防止灾难性遗忘

---

## 技术细节

### vLLM API调用

```python
payload = {
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": messages,
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {
        "enable_thinking": true  # 启用thinking模式
    }
}
```

### 提取流程

```python
# 1. 统一提取
from utils.thinking_extraction import extract_thinking_and_code_unified
parsed = extract_thinking_and_code_unified(response)

# 2. 获取组件
thinking = parsed['thinking']  # 用于SFT训练
code = parsed['code']          # 用于执行/奖励
```

### SFT数据准备

```python
from utils.sft_data import prepare_teacher_sft_batch

batch = prepare_teacher_sft_batch(
    teacher_examples=[{
        "prompt": "问题描述",
        "thinking": "推理过程",
        "code": "代码"
    }],
    tokenizer=tokenizer,
    include_thinking=True
)

# batch["labels"]:
# - Prompt部分: -100
# - Response部分: token_ids
```

---

## 预期结果

### 训练后(3轮)

| 指标 | 分数 | 说明 |
|------|------|------|
| pass@1 | 45-50% | 单次采样成功率 |
| pass@5 | 65-70% | 5次采样成功率 |
| pass@10 | 75-80% | 10次采样成功率 |

### 与基线比较

- vs 标准SFT: +10-15% (pass@1)
- vs 仅GRPO: +5-8% (pass@1)
- vs 仅Teacher-SFT: +8-12% (pass@1)

---

## 许可证

MIT License

---

## 重要提示

⚠️ **训练前必须运行**: `python test_all.py`
⚠️ **所有阶段使用统一提示词**: `get_unified_prompt()`
⚠️ **验证提取流程**: thinking和代码分别提取
⚠️ **检查SFT标签**: Prompt被mask为-100
⚠️ **RL优化完整序列**: thinking + code一起优化

---

**问题或建议?** 检查test_all.py的输出,或查看代码中的注释。
