from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger

from generative_recs.config import load_config, save_config
from generative_recs.preparation import build_splits, generate_embeddings, load_items, load_reviews


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the preparation YAML file.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Directory to create for this preparation run; must not already exist.",
        ),
    ],
) -> None:
    """Prepare interaction splits, item text, and embeddings."""
    config = load_config(config_path, "preparation")

    output_dir.mkdir(parents=True, exist_ok=False)

    logger.add(output_dir / "run.log", level="INFO")
    logger.info(f"Output directory: {output_dir}")

    configs_dir = output_dir / "configs"
    configs_dir.mkdir()

    save_config(config, configs_dir / "preparation.yaml")

    reviews = load_reviews(config.reviews_path)
    splits = build_splits(reviews)
    items = load_items(config.metadata_path, set(reviews["asin"]))

    logger.info(
        f"Prepared {len(reviews):,} reviews, {len(splits):,} users, and {len(items):,} items."
    )

    embeddings = generate_embeddings(items, config.embedding)

    splits.to_parquet(output_dir / "splits.parquet", index=False)
    items.to_parquet(output_dir / "items.parquet", index=False)
    torch.save(embeddings, output_dir / "embeddings.pt")

    logger.info(f"Preparation complete. Saved outputs to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
