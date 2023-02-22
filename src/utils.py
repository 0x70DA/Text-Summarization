import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers.utils import cached_property, is_tf_available, requires_backends

logger = logging.getLogger(__name__)

if is_tf_available():
    import tensorflow as tf


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    input_column: str = field(
        metadata={
            "help": "The name of the column in the datasets containing the inputs."
        },
    )
    target_column: str = field(
        metadata={
            "help": "The name of the column in the datasets containing the targets."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TFTrainingArguments:
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    use_gpu: bool = field(
        default=True, metadata={"help": "Whether or not to train on GPU."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )

    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    no_cuda: bool = field(
        default=False, metadata={"help": "Do not use CUDA even when it is available"}
    )
    xla: bool = field(
        default=False,
        metadata={"help": "Whether to activate the XLA compilation or not"},
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )
    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    push_to_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository to which push the model."},
    )
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    save_local: bool = field(
        default=False,
        metadata={"help": "Whether or not to save a local copy of the model."},
    )

    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        requires_backends(self, ["tf"])
        logger.info("Tensorflow: setting up strategy")

        gpus = tf.config.list_physical_devices("GPU")

        if self.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            if len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                strategy = tf.distribute.MirroredStrategy()
            else:
                raise ValueError(
                    "Cannot find the proper strategy, please check your environment properties."
                )

        return strategy

    @property
    def strategy(self) -> "tf.distribute.Strategy":
        """
        The strategy used for distributed training.
        """
        requires_backends(self, ["tf"])
        return self._setup_strategy

    @property
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        requires_backends(self, ["tf"])
        return self._setup_strategy.num_replicas_in_sync
