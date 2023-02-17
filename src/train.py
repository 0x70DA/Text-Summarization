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
from package.utils import ModelArguments, DataTrainingArguments

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
        model_args: ModelArguments; data_args:DataTrainingArguments; training_args: TFTrainingArguments = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args: ModelArguments; data_args:DataTrainingArguments; training_args: TFTrainingArguments = parser.parse_args_into_dataclasses()
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

main()