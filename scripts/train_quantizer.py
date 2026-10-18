import json
from pathlib import Path
from shutil import copy2
from typing import Annotated

import torch
import typer
from loguru import logger

from generative_recs.config import load_config, save_config
from generative_recs.quantization import (
    evaluate_quantizer,
    generate_semantic_ids,
    train_quantizer,
)
from generative_recs.rqvae import RQVAE
from generative_recs.utils import get_device, set_seed


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to quantization YAML file.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Directory to create for this quantizer run; must not already exist.",
        ),
    ],
) -> None:
    """Train quantizer and generate Semantic IDs from its best checkpoint."""
    config = load_config(config_path, "quantization")

    output_dir.mkdir(parents=True, exist_ok=False)

    logger.add(output_dir / "run.log", level="INFO")
    logger.info(f"Output directory: {output_dir}")

    configs_dir = output_dir / "configs"
    configs_dir.mkdir()
    save_config(config, configs_dir / "quantization.yaml")
    copy2(
        config.preparation_dir / "configs" / "preparation.yaml",
        configs_dir / "preparation.yaml",
    )

    set_seed(config.seed)
    device = get_device()

    embeddings_dict = torch.load(
        config.preparation_dir / "embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )
    asins = list(embeddings_dict)
    embeddings = torch.stack([embeddings_dict[asin] for asin in asins]).float()

    model = RQVAE(**config.model.model_dump()).to(device)

    logger.info(f"Training on {len(asins):,} items for up to {config.training.max_steps:,} steps.")
    train_quantizer(model, embeddings, config.training, output_dir)

    metrics = evaluate_quantizer(
        model,
        embeddings,
        config.training.batch_size,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    asin_to_sid = generate_semantic_ids(model, asins, embeddings, config.training.batch_size)
    torch.save(
        {
            "asin_to_sid": asin_to_sid,
            "sid_to_asin": {v: k for k, v in asin_to_sid.items()},
        },
        output_dir / "mappings.pt",
    )

    logger.info(f"Quantization complete. Saved outputs to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
