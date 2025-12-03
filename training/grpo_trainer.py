"""
Custom GRPO Trainer with LoRA adapters and Dual-Source objective
"""
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from .losses import (
    compute_grpo_loss,
    compute_teacher_sft_loss,
    compute_self_sft_loss,
    compute_dual_source_loss,
    compute_advantages,
)
from .utils import (
    set_seed,
    print_trainable_parameters,
    prepare_model_for_kbit_training,
    gather_rollout_stats,
    format_prompt_for_chat,
    compute_response_log_probs,
    create_reference_model,
    save_checkpoint,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.code_executor import compute_reward


class GRPOTrainer:
    """Custom GRPO Trainer with LoRA and Dual-Source objective"""

    def __init__(self, config):
        """
        Initialize GRPO Trainer

        Args:
            config: MasterConfig object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seed
        set_seed(config.training.seed)

        # Initialize models
        print("Initializing models...")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.ref_model = None  # Will be created after LoRA setup

        # Apply LoRA
        print("Applying LoRA adapters...")
        self._apply_lora()

        # Create reference model (frozen copy before LoRA)
        print("Creating reference model...")
        self.ref_model = self._create_reference_model()

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None  # Will be created in train()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Teacher data (will be loaded in train())
        self.teacher_data = None

        # Logging
        self.use_wandb = config.training.report_to == "wandb"
        if self.use_wandb:
            import wandb
            wandb.init(
                project="teacher-grpo",
                name=config.training.run_name or "grpo-training",
                config=vars(config),
            )

    def _load_tokenizer(self):
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.student.model_name,
            trust_remote_code=self.config.student.trust_remote_code,
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _load_model(self):
        """Load student model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.student.model_name,
            torch_dtype=self.config.student.torch_dtype,
            device_map=self.config.student.device_map,
            trust_remote_code=self.config.student.trust_remote_code,
            load_in_8bit=self.config.student.load_in_8bit,
            load_in_4bit=self.config.student.load_in_4bit,
        )

        # Prepare for training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.training.gradient_checkpointing
        )

        return model

    def _apply_lora(self):
        """Apply LoRA adapters to model"""
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        print_trainable_parameters(self.model)

    def _create_reference_model(self):
        """Create frozen reference model for KL penalty"""
        # Load base model without LoRA
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.student.model_name,
            torch_dtype=self.config.student.torch_dtype,
            device_map=self.config.student.device_map,
            trust_remote_code=self.config.student.trust_remote_code,
        )

        # Freeze all parameters
        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model.eval()
        return ref_model

    def _create_optimizer(self):
        """Create optimizer"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def generate_responses(
        self,
        queries: List[str],
        num_samples: int = 4,
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Generate responses from current policy

        Args:
            queries: List of prompts
            num_samples: Number of samples per prompt

        Returns:
            Tuple of (responses, input_ids, attention_masks)
        """
        self.model.eval()

        all_responses = []
        all_input_ids = []
        all_attention_masks = []

        with torch.no_grad():
            for query in tqdm(queries, desc="Generating responses"):
                query_responses = []
                query_input_ids = []
                query_attention_masks = []

                # Tokenize query
                query_tokens = self.tokenizer(
                    query,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                ).to(self.device)

                # Generate multiple samples
                for _ in range(num_samples):
                    outputs = self.model.generate(
                        **query_tokens,
                        max_new_tokens=self.config.grpo.max_new_tokens,
                        temperature=self.config.grpo.temperature,
                        top_p=self.config.grpo.top_p,
                        top_k=self.config.grpo.top_k,
                        do_sample=self.config.grpo.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode response
                    full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = self.tokenizer.decode(
                        outputs[0][query_tokens.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )

                    query_responses.append(response)
                    query_input_ids.append(outputs[0])

                    # Create attention mask
                    attention_mask = torch.ones_like(outputs[0])
                    query_attention_masks.append(attention_mask)

                all_responses.append(query_responses)
                all_input_ids.append(query_input_ids)
                all_attention_masks.append(query_attention_masks)

        return all_responses, all_input_ids, all_attention_masks

    def compute_rewards(
        self,
        responses: List[List[str]],
        tasks: List[Dict],
    ) -> List[List[float]]:
        """
        Compute rewards by executing code against test cases

        Args:
            responses: Generated code responses
            tasks: Task dictionaries with test cases

        Returns:
            List of reward lists
        """
        all_rewards = []

        for task_responses, task in zip(responses, tasks):
            task_rewards = []

            for response in task_responses:
                reward = compute_reward(
                    code=response,
                    test_cases=task["test_list"],
                    setup_code=task.get("test_setup_code", ""),
                    timeout=5,
                    reward_type="binary",  # Can be configured
                )
                task_rewards.append(reward)

            all_rewards.append(task_rewards)

        return all_rewards

    def compute_log_probs(
        self,
        model,
        input_ids_list: List[List[torch.Tensor]],
        attention_masks_list: List[List[torch.Tensor]],
        query_lengths: List[int],
    ) -> List[List[float]]:
        """
        Compute log probabilities for responses

        Args:
            model: Model to use
            input_ids_list: List of input ID lists
            attention_masks_list: List of attention mask lists
            query_lengths: Query lengths

        Returns:
            List of log probability lists
        """
        all_log_probs = []

        for input_ids_group, attention_masks_group, query_len in zip(
            input_ids_list, attention_masks_list, query_lengths
        ):
            group_log_probs = []

            for input_ids, attention_mask in zip(input_ids_group, attention_masks_group):
                # Prepare batch
                input_ids = input_ids.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)

                # Forward pass
                with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                # Compute log probs
                log_probs = compute_response_log_probs(
                    logits=outputs.logits,
                    input_ids=input_ids,
                    query_length=query_len,
                )

                group_log_probs.append(log_probs.item())

            all_log_probs.append(group_log_probs)

        return all_log_probs

    def train_step(
        self,
        batch: Dict,
        teacher_batch: Optional[Dict] = None,
        successful_samples: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Perform single training step with Dual-Source objective

        Args:
            batch: GRPO batch data
            teacher_batch: Teacher SFT batch (optional)
            successful_samples: Successful samples for Self-SFT (optional)

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Compute GRPO loss
        current_log_probs = batch["log_probs"]
        ref_log_probs = batch["ref_log_probs"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]

        grpo_loss, grpo_metrics = compute_grpo_loss(
            log_probs=current_log_probs,
            ref_log_probs=ref_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            kl_coef=self.config.grpo.kl_coef,
            clip_range=self.config.grpo.clip_range,
        )

        # Compute Teacher-SFT loss
        teacher_sft_loss = None
        if teacher_batch is not None:
            teacher_sft_loss = compute_teacher_sft_loss(
                model=self.model,
                teacher_input_ids=teacher_batch["input_ids"],
                teacher_attention_mask=teacher_batch["attention_mask"],
                teacher_labels=teacher_batch["labels"],
            )

        # Compute Self-SFT loss
        self_sft_loss = None
        if successful_samples:
            self_sft_loss = compute_self_sft_loss(
                model=self.model,
                successful_samples=successful_samples,
                tokenizer=self.tokenizer,
                top_k=self.config.dual_source.self_sft_top_k,
                min_reward=self.config.dual_source.self_sft_min_reward,
                max_length=self.config.dual_source.self_sft_max_length,
            )

        # Compute Dual-Source composite loss
        total_loss, loss_metrics = compute_dual_source_loss(
            grpo_loss=grpo_loss,
            teacher_sft_loss=teacher_sft_loss,
            self_sft_loss=self_sft_loss,
            grpo_weight=self.config.dual_source.grpo_weight,
            teacher_sft_weight=self.config.dual_source.teacher_sft_weight,
            self_sft_weight=self.config.dual_source.self_sft_weight,
        )

        # Backward pass
        (total_loss / self.config.training.gradient_accumulation_steps).backward()

        # Combine metrics
        metrics = {**grpo_metrics, **loss_metrics}

        return metrics

    def train(
        self,
        train_data: List[Dict],
        teacher_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
    ):
        """
        Main training loop

        Args:
            train_data: Training task data
            teacher_data: Teacher synthetic traces
            val_data: Validation data (optional)
        """
        self.teacher_data = teacher_data

        # Calculate training steps
        num_epochs = self.config.training.num_train_epochs
        batch_size = self.config.training.per_device_train_batch_size
        grad_accum_steps = self.config.training.gradient_accumulation_steps

        steps_per_epoch = len(train_data) // (batch_size * grad_accum_steps)
        total_steps = steps_per_epoch * num_epochs

        # Create scheduler
        self.scheduler = self._create_scheduler(total_steps)

        print(f"\n{'=' * 50}")
        print(f"Training Configuration:")
        print(f"{'=' * 50}")
        print(f"Total epochs: {num_epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Training samples: {len(train_data)}")
        print(f"Teacher samples: {len(teacher_data)}")
        print(f"{'=' * 50}\n")

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_metrics = []

            # Shuffle data
            import random
            random.shuffle(train_data)

            # Process in batches
            for batch_idx in tqdm(range(0, len(train_data), batch_size), desc="Training"):
                batch_tasks = train_data[batch_idx:batch_idx + batch_size]

                # Generate responses using UNIFIED prompt format
                # Import here to avoid circular dependency
                from utils.prompts import get_unified_prompt

                queries = [
                    format_prompt_for_chat(
                        self.tokenizer,
                        get_unified_prompt(task['prompt'], dataset_type=task.get('dataset', 'mbpp'))
                    )
                    for task in batch_tasks
                ]

                responses, input_ids_list, attention_masks_list = self.generate_responses(
                    queries,
                    num_samples=self.config.grpo.num_samples_per_prompt,
                )

                # Compute rewards
                rewards_list = self.compute_rewards(responses, batch_tasks)

                # Prepare GRPO batch
                # (Simplified - full implementation would handle this more elegantly)

                # Perform training step every gradient_accumulation_steps
                if (self.global_step + 1) % grad_accum_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Log metrics
                    if self.use_wandb:
                        import wandb
                        wandb.log(epoch_metrics[-1] if epoch_metrics else {}, step=self.global_step)

                self.global_step += 1

                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    save_checkpoint(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        output_dir=self.config.training.output_dir,
                        epoch=epoch,
                        step=self.global_step,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )

        # Save final model
        print("\nSaving final model...")
        final_dir = Path(self.config.training.output_dir) / "final_model"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"✓ Training complete! Model saved to {final_dir}")
