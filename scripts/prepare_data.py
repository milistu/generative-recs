from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger

from generative_recs.config import load_preparation_config, save_config
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
            help="Parent directory for timestamped preparation runs.",
        ),
    ],
) -> None:
    """Prepare interaction splits, item text, and embeddings."""
    config = load_preparation_config(config_path)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir.resolve() / f"{config_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger.add(run_dir / "run.log", level="INFO")
    logger.info(f"Output directory: {run_dir}")

    configs_dir = run_dir / "configs"
    configs_dir.mkdir()

    save_config(config, configs_dir / "preparation.yaml")

    reviews = load_reviews(config.reviews_path)
    splits = build_splits(reviews)
    items = load_items(config.metadata_path, set(reviews["asin"]))

    logger.info(
        f"Prepared {len(reviews):,} reviews, {len(splits):,} users, and {len(items):,} items."
    )

    embeddings = generate_embeddings(items, config.embedding)

    splits.to_parquet(run_dir / "splits.parquet", index=False)
    items.to_parquet(run_dir / "items.parquet", index=False)
    torch.save(embeddings, run_dir / "embeddings.pt")

    logger.info(f"Preparation complete. Saved outputs to {run_dir}")


if __name__ == "__main__":
    typer.run(main)
