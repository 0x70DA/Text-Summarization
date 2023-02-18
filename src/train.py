import json
import os
import sys
import logging

import datasets
import evaluate
import nltk
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from utils import ModelArguments, DataTrainingArguments

import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TFAutoModelForSeq2SeqLM,
    HfArgumentParser,
    KerasMetricCallback,
    PushToHubCallback,
    TFTrainingArguments,
    create_optimizer,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)
nltk.download("punkt")

def main():
    # region Argument parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse arguments from json file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)

    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region T5 special-casing
    if data_args.source_prefix is None and model_args.model_name in [
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

    # region Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # endregion

    set_seed(training_args.seed)

    # region Load datasets
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token
    )
    # endregion

    # region Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
        use_fast_toeknizer=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token
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

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
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
    with training_args.main_process_first(desc="Train dataset map preprocessing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset"
        )
    
    if training_args.do_eval:
        # max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    else:
        eval_dataset = None
    
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

    with training_args.strategy.scope():
        # region Prepare model
        model = TFAutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token
        )
        # endregion

        # region Prepare TF datasets

        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=128,  # Reduce the number of unique shapes for XLA, especially for generation
            return_tensors="tf",
        )

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
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

        # region Metric and KerasMetricCallback
        if training_args.do_eval:
            metric = evaluate.load("rouge")

            if data_args.val_max_target_length is None:
                data_args.val_max_target_length = data_args.max_target_length

            gen_kwargs = {
                "max_length": data_args.val_max_target_length,
                "num_beams": data_args.num_beams,
                "no_repeat_ngram_size": 0,  # Not supported under XLA right now, and some models set it by default
            }

            def compute_metrics(predictions, labels):
                # if isinstance(predictions, tuple):
                #     predictions = predictions[0]
                decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metrics = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                metrics = {key: round(val * 100, 4) for key, val in metrics.items()}
                return metrics

            metric_callback = KerasMetricCallback(
                metric_fn=compute_metrics,
                eval_dataset=tf_eval_dataset,
                predict_with_generate=True,
                use_xla_generation=True,
                generate_kwargs=gen_kwargs,
            )
            callbacks = [metric_callback]
        else:
            callbacks = []
        # endregion

        # region Preparing push_to_hub and model card
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name.split("/")[-1]
        if not push_to_hub_model_id:
            if data_args.dataset_name is not None:
                push_to_hub_model_id = f"{model_name}-finetuned-{data_args.dataset_name}"
            else:
                push_to_hub_model_id = f"{model_name}-finetuned-summarization"

        model_card_kwargs = {"finetuned_from": model_args.model_name, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            model_card_kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                model_card_kwargs["dataset_args"] = data_args.dataset_config_name
                model_card_kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                model_card_kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            # Because this training can be quite long, we save once per epoch.
            callbacks.append(
                PushToHubCallback(
                    output_dir=training_args.output_dir,
                    model_id=push_to_hub_model_id,
                    organization=training_args.push_to_hub_organization,
                    token=training_args.push_to_hub_token,
                    tokenizer=tokenizer,
                    **model_card_kwargs,
                )
            )
        # endregion

        # region Training
        model.compile(optimizer=optimizer, jit_compile=training_args.xla)
        eval_metrics = None

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        logger.info(f"  Total optimization steps = {num_train_steps}")

        if training_args.xla and not data_args.pad_to_max_length:
            logger.warning(
                "XLA training may be slow at first when --pad_to_max_length is not set "
                "until all possible shapes have been compiled."
            )
        history = model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=int(training_args.num_train_epochs), callbacks=callbacks)
        eval_metrics = {key: val[-1] for key, val in history.history.items()}
        # endregion

        # region Evaluation

        if training_args.do_eval:
            logger.info("Evaluation...")

            @tf.function(jit_compile=True)
            def generate(**kwargs):
                return model.generate(**kwargs)

            for batch, labels in tf_eval_dataset:
                batch.update(gen_kwargs)
                generated_tokens = generate(**batch)
                # if isinstance(generated_tokens, tuple):
                #     generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            eval_metrics = metric.compute(use_stemmer=True)

            result = {key: round(val * 100, 4) for key, val in eval_metrics.items()}
            logger.info(result)
        # endregion

        if training_args.output_dir is not None and eval_metrics is not None:
            output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(eval_metrics))

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
