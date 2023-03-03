import json
import logging
import os
import sys

import evaluate
from huggingface_hub import login
import nltk
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModelForSeq2SeqLM,
    create_optimizer,
)
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils import DataTrainingArguments, ModelArguments, TFTrainingArguments

logger = logging.getLogger(__name__)
logger.info(f"Number of available GPUs: {len(tf.config.list_physical_devices('GPU'))}")

nltk.download("punkt")


def main():
    # region Argument parsing
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TFTrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TFTrainingArguments

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse arguments from json file
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create the output directory if it doesn't exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    if training_args.use_gpu and len(tf.config.list_physical_devices("GPU")) == 0:
        raise ValueError("No GPUs available for training")

    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Login to HuggingFace hub
    if training_args.push_to_hub_token is not None:
        login(training_args.push_to_hub_token, add_to_git_credential=True)

    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region Setup wandb
    callbacks = []
    if data_args.wandb_track:
        wandb.init(project="Summarization")
        callbacks.extend(
            [WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
        )

    # endregion

    # region T5 special-casing
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # endregion

    # region Load dataset
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token,
    )
    # endregion

    # region Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=data_args.max_source_length,
        use_fast_tokenizer=model_args.use_fast_tokenizer,
        use_auth_token=model_args.use_auth_token,
    )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    # endregion

    # region Data preprocessing
    column_names = raw_datasets["train"].column_names
    input_column = data_args.input_column
    target_column = data_args.target_column
    if input_column not in column_names:
        raise ValueError(
            f"--input_column value `{data_args.input_column} needs to be one of: {', '.join(column_names)}`"
        )
    if target_column not in column_names:
        raise ValueError(
            f"--target_column value `{data_args.target_column} needs to be one of: {', '.join(column_names)}`"
        )

    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(batch):
        inputs = batch[input_column]
        inputs = [prefix + s for s in inputs]
        targets = batch[target_column]

        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=data_args.max_target_length,
            padding=padding,
            truncation=True,
        )

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Preprocess train dataset
    if "train" not in raw_datasets:
        raise ValueError("A train dataset is required.")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    # Preprocess validation dataset
    if "validation" not in raw_datasets:
        raise ValueError("A validation dataset is required.")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
    # endregion

    # region Text preprocessing
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    # endregion

    # region Compute metric
    rouge_score = evaluate.load("rouge")

    def compute_metric(
        model,
        tokenized_dataset=eval_dataset,
        metric=rouge_score,
        batch_size=8,
    ):
        generation_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            return_tensors="tf",
        )

        tf_generate_dataset = model.prepare_tf_dataset(
            tokenized_dataset,
            collate_fn=generation_data_collator,
            shuffle=False,
            batch_size=batch_size,
            drop_remainder=True,
        )

        @tf.function(jit_compile=True)
        def generate_with_xla(batch):
            return model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
            )

        all_preds = []
        all_labels = []
        for batch, labels in tqdm(tf_generate_dataset):
            predictions = generate_with_xla(batch)
            decoded_preds = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            labels = labels.numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

        return metric.compute(
            predictions=all_preds, references=all_labels, use_stemmer=True
        )

        # endregion

    with training_args.strategy.scope():
        # region Load model
        model = TFAutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.use_auth_token,
        )
        # endregion

        # region Prepare TF datasets
        label_pad_token_id = (
            -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            return_tensors="tf",
        )

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.AUTO
        )

        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = (
            training_args.per_device_train_batch_size * num_replicas
        )
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        tf_train_dataset = model.prepare_tf_dataset(
            train_dataset,
            collate_fn=data_collator,
            batch_size=total_train_batch_size,
            shuffle=True,
        ).with_options(dataset_options)

        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=total_eval_batch_size,
            shuffle=False,
        ).with_options(dataset_options)
        # endregion

        # region Optimizer, loss and LR scheduling
        num_train_steps = int(len(tf_train_dataset) * training_args.num_train_epochs)
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0

        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
            adam_global_clipnorm=training_args.max_grad_norm,
        )
        # endregion

        # region Training
        model.compile(optimizer=optimizer, jit_compile=training_args.xla)

        if training_args.fp16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        eval_metrics = None

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
        )
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        logger.info(f"  Total optimization steps = {num_train_steps}")

        if training_args.xla and not data_args.pad_to_max_length:
            logger.warning(
                "XLA training may be slow at first when --pad_to_max_length is not set "
                "until all possible shapes have been compiled."
            )

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        )

        history = model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            callbacks=callbacks,
        )

        train_metrics = {key: val[-1] for key, val in history.history.items()}
        train_result = {key: val for key, val in train_metrics.items()}
        logger.info(train_result)

        train_result_file = os.path.join(training_args.output_dir, "train_results.json")
        with open(train_result_file, "w") as writer:
            writer.write(json.dumps(train_metrics))
        # endregion

        # region Evaluation
        if training_args.do_eval:
            logger.info("Evaluation...")

            eval_metrics = compute_metric(model)

            eval_result = {
                key: round(val * 100, 4) for key, val in eval_metrics.items()
            }
            logger.info(eval_result)

        if eval_metrics is not None:
            eval_result_file = os.path.join(
                training_args.output_dir, "eval_results.json"
            )
            with open(eval_result_file, "w") as writer:
                writer.write(json.dumps(eval_metrics))
        # endregion

    # Training is done. Save the model.

    # region Push to hub

    if training_args.push_to_hub and training_args.push_to_hub_token is not None:
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name_or_path.split("/")[-1]
        if not push_to_hub_model_id:
            if data_args.dataset_name is not None:
                push_to_hub_model_id = (
                    f"{model_name}-finetuned-{data_args.dataset_name.split('/')[-1]}"
                )
            else:
                push_to_hub_model_id = f"{model_name}-finetuned-summarization"

        model.push_to_hub(
            push_to_hub_model_id, use_auth_token=training_args.push_to_hub_token
        )
        tokenizer.push_to_hub(
            push_to_hub_model_id, use_auth_token=training_args.push_to_hub_token
        )

    # endregion

    if training_args.save_local:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    if data_args.wandb_track:
        wandb.finish()


if __name__ == "__main__":
    main()
