"""Script for a training run."""

import hydra

import os
import logging
from transformers import TrainerCallback

from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from normfree_transformers import model_utils, train_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="normfree_transformers/config", config_name="config")
def launch(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    transformers.set_seed(cfg.seed)
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid", split="validation"
    )

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=0).select(
                range(cfg.num_token_mult * 100000)
            ),
            "valid": ds_valid.shuffle(seed=0).select(range(2000)),
        }
    )

    context_length = cfg.model.context_length
    tokenizer = AutoTokenizer.from_pretrained(
        "huggingface-course/code-search-net-tokenizer", use_fast=True
    )

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    model_config = AutoConfig.from_pretrained(
        cfg.model.name,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        n_inner=int(cfg.model.n_embd * cfg.model.mlp_width_mult),
        initializer_range=cfg.model.initializer_range,
        output_attentions=cfg.report_attn_entropy,
        
    )
    model = GPT2LMHeadModel(model_config)

    model_config.update(
        {                           
            "norm_type": cfg.model.norm_type,
            "activation_function": cfg.model.activation_function,
            "lrelu_neg_slope": cfg.model.lrelu_neg_slope,
            "learnable_lrelu_mode": cfg.model.learnable_lrelu_mode,                        
        }
    )

    model = model_utils.convertGPT2model(model, model_config)
    print (model)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=cfg.train.device_train_batch_size,
        per_device_eval_batch_size=cfg.train.device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        learning_rate=cfg.train.learning_rate,
        save_steps=10000,
        fp16=True,
        report_to="wandb" if cfg.use_wandb else "none",
        adam_epsilon=cfg.train.adam_epsilon,
        max_grad_norm=cfg.train.max_grad_norm,
    )
    
    args.report_attn_entropy = cfg.report_attn_entropy
    args.report_nan_counts = cfg.report_nan_counts
    args.report_neg_slope = cfg.report_neg_slope
    
    trainer = train_utils.MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    # Initialize logging callbacks for entropy
    entropy_callback = train_utils.EntropyLoggingCallback( 
        output_dir="entropy_logs",
        num_layers=trainer.model.config.n_layer
    )

    trainer.entropy_logging_callback = entropy_callback  
    trainer.add_callback(entropy_callback)
    
    # Initialize logging callbacks for NaN counts
    nan_count_callback = train_utils.NaNCountLoggingCallback(
        output_dir="nan_count_logs",
        num_layers=trainer.model.config.n_layer
    )
    
    trainer.nan_count_logging_callback = nan_count_callback  
    trainer.add_callback(nan_count_callback)

    # Initialize logging callbacks for negative slopes in learnable leaky relu
    neg_slope_callback = train_utils.SlopeLoggingCallback(
        cfg.model.learnable_lrelu_mode, 
        output_dir="neg_slope_logs",
        num_layers=trainer.model.config.n_layer
    )

    trainer.neg_slope_logging_callback = neg_slope_callback  
    trainer.add_callback(neg_slope_callback)
    
    trainer.train()


if __name__ == "__main__":
    launch()
