from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


class EmbeddingConfig(BaseModel):
    """Text encoder and batching settings for item embeddings."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    max_seq_length: PositiveInt
    batch_size: PositiveInt


class PreparationConfig(BaseModel):
    """Source files and embedding settings for data preparation."""

    model_config = ConfigDict(extra="forbid")

    reviews_path: Path
    metadata_path: Path
    embedding: EmbeddingConfig


class QuantizationModelConfig(BaseModel):
    """RQ-VAE architecture and codebook settings."""

    model_config = ConfigDict(extra="forbid")

    input_dim: PositiveInt
    hidden_dims: list[PositiveInt] = Field(min_length=1)
    latent_dim: PositiveInt
    num_quantizers: PositiveInt
    codebook_size: PositiveInt
    use_cosine_sim: bool
    kmeans_init: bool
    kmeans_iters: PositiveInt
    commitment_weight: NonNegativeFloat
    reconstruction_loss: Literal["cosine", "squared_l2"] = "cosine"
    rotation_trick: bool = True
    codebook_update: Literal["ema", "gradient"] = "ema"


class QuantizationTrainingConfig(BaseModel):
    """Quantizer optimization, evaluation, and stopping settings."""

    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    max_steps: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    evaluate_every: PositiveInt
    early_stopping_min_steps: PositiveInt
    early_stopping_usage: float = Field(ge=0.0, le=1.0)
    early_stopping_patience: PositiveInt | None
    early_stopping_min_delta: NonNegativeFloat


class QuantizationConfig(BaseModel):
    """Prepared data and settings for a quantizer training run."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0, lt=2**32)
    preparation_dir: Path
    model: QuantizationModelConfig
    training: QuantizationTrainingConfig


class DataConfig(BaseModel):
    """Dataset paths and Semantic ID encoding settings."""

    model_config = ConfigDict(extra="forbid")

    splits_path: Path
    mappings_path: Path
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

    optimizer: Literal["adamw", "adafactor"] = "adamw"
    adafactor_scale_parameter: bool = True
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


def load_config(
    path: Path, config_type: Literal["preparation", "quantization", "retrieval"]
) -> PreparationConfig | QuantizationConfig | RetrievalConfig:
    """Load YAML and validate configuration settings.

    Args:
        path: Path to YAML configuration file.
        config_type: Type of configuration to load.

    Raises:
        OSError: The configuration file cannot be read.
        yaml.YAMLError: The file contains invalid YAML.
        pydantic.ValidationError: Settings fail schema validation.
    """
    with open(path, encoding="utf-8") as file:
        values = yaml.safe_load(file)

    match config_type:
        case "preparation":
            return PreparationConfig.model_validate(values)
        case "quantization":
            return QuantizationConfig.model_validate(values)
        case "retrieval":
            return RetrievalConfig.model_validate(values)
        case _:
            raise ValueError(f"Invalid configuration type: {config_type}")


def save_config(config: BaseModel, path: Path) -> None:
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
