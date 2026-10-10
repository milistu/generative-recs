from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from loguru import logger
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

from generative_recs.config import load_config
from generative_recs.dataset import TigerDataset, custom_collate
from generative_recs.evaluation import compute_metrics
from generative_recs.utils import get_device, set_seed


def main(
    run_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Run directory containing configs/retrieval.yaml and best/.",
        ),
    ],
    split: Annotated[
        Literal["val", "test"], typer.Option(help="Dataset split to evaluate.")
    ] = "val",
) -> None:
    """Evaluate a run's selected model and save its retrieval metrics."""
    run_dir = run_dir.resolve()
    config = load_config(run_dir / "configs" / "retrieval.yaml")

    set_seed(config.seed)
    device = get_device()

    dataset = TigerDataset(**config.data.model_dump(), split=split, seed=config.seed)
    dataset.samples = dataset.samples[: config.evaluation.max_samples]

    model = T5ForConditionalGeneration.from_pretrained(run_dir / "best").to(device)

    sid_data = torch.load(
        config.data.semantic_ids_path,
        map_location="cpu",
        weights_only=False,
    )

    collate_fn = partial(custom_collate, pad_token_id=dataset.pad_token)
    retrieval_metrics_fn = partial(
        compute_metrics,
        num_levels=dataset.num_levels,
        beam_size=model.generation_config.num_return_sequences,
        codebook_size=config.data.codebook_size,
        sid_to_asin=sid_data["sid_to_asin"],
        at_k=config.evaluation.at_k,
    )

    evaluation_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_eval_batch_size=config.evaluation.batch_size,
        predict_with_generate=True,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,
        dataloader_pin_memory=device.type == "cuda",
        seed=config.seed,
        report_to="none",
    )

    evaluator = Seq2SeqTrainer(
        model=model,
        args=evaluation_args,
        eval_dataset=dataset,
        data_collator=collate_fn,
        compute_metrics=retrieval_metrics_fn,
    )

    logger.info(f"Evaluating {split} on {len(dataset):,} examples from {run_dir}")

    metrics = evaluator.evaluate(metric_key_prefix=split)
    metrics[f"{split}_samples"] = len(dataset)
    evaluator.save_metrics(split, metrics, combined=False)

    logger.info(f"Results: {metrics}")
    logger.info(f"Saved metrics to `{run_dir / f'{split}_results.json'}`")


if __name__ == "__main__":
    typer.run(main)
