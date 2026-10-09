from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


class DataConfig(BaseModel):
    """Dataset paths and Semantic ID encoding settings."""

    model_config = ConfigDict(extra="forbid")

    splits_path: Path
    semantic_ids_path: Path
    max_seq_len: PositiveInt
    sliding_window: bool
    codebook_size: PositiveInt
    num_levels: PositiveInt
    num_user_buckets: PositiveInt


class ModelConfig(BaseModel):
    """Generative model architecture."""

    model_config = ConfigDict(extra="forbid")

    d_model: PositiveInt
    d_kv: PositiveInt
    d_ff: PositiveInt
    num_layers: PositiveInt
    num_heads: PositiveInt
    dropout_rate: float = Field(ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Optimiser, schedule, and checkpoint settings."""

    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    max_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    constant_lr_steps: PositiveInt
    max_grad_norm: NonNegativeFloat
    logging_steps: PositiveInt
    save_total_limit: PositiveInt
    max_samples: PositiveInt | None = None


class EvaluationConfig(BaseModel):
    """Evaluation frequency, candidate generation, and metric settings."""

    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    every_steps: PositiveInt
    beam_size: PositiveInt
    at_k: tuple[PositiveInt, ...] = Field(min_length=1)
    selection_metric: str
    max_samples: PositiveInt | None = None


class RetrievalConfig(BaseModel):
    """Complete configuration for a retrieval experiment."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0, lt=2**32)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> RetrievalConfig:
    """Load YAML and validate retrieval settings.

    Args:
        path: Path to YAML configuration file.

    Raises:
        OSError: The configuration file cannot be read.
        yaml.YAMLError: The file contains invalid YAML.
        pydantic.ValidationError: Settings fail schema validation.
    """
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file)

    return RetrievalConfig.model_validate(values)


def save_config(config: RetrievalConfig, path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Path to the YAML file to save the configuration.
    """
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            stream=file,
            sort_keys=False,
        )
