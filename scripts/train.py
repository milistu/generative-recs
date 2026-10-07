from pathlib import Path
from typing import Annotated
from functools import partial

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from loguru import logger

import typer
from generative_recs.config import load_config, save_config
from generative_recs.utils import set_seed, get_device, get_num_params
from generative_recs.dataset import TigerDataset, custom_collate
from generative_recs.model import create_model
from generative_recs.evaluation import compute_metrics


def inv_sqrt_schedule(step: int, constant_lr_steps: int) -> float:
    """Keep the learning rate multiplier at 1, then decay as inverse square root."""
    if step < constant_lr_steps:
        return 1.0
    return (constant_lr_steps / step) ** 0.5


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the experiment YAML file.",
        ),
    ],
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            file_okay=False,
            help="Directory for this run's configuration, logs, and checkpoints.",
        ),
    ],
) -> None:
    """Train the retrieval model and export the best validation checkpoint."""

    config = load_config(config_path)

    run_dir = run_dir.resolve()

    if run_dir.exists() and any(run_dir.iterdir()):
        raise typer.BadParameter(
            "Use a new or empty directory for a new run.",
            param_hint="--run-dir",
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "run.log", level="INFO")
    save_config(config, run_dir / "config.yaml")
    logger.info(f"Run directory: {run_dir}")

    set_seed(config.seed)
    device = get_device()

    dataset_kwargs = config.data.model_dump()

    train_dataset = TigerDataset(**dataset_kwargs, split="train", seed=config.seed)
    val_dataset = TigerDataset(**dataset_kwargs, split="val", seed=config.seed)
    train_dataset.samples = train_dataset.samples[: config.training.max_samples]
    val_dataset.samples = val_dataset.samples[: config.evaluation.max_samples]

    collate_fn = partial(custom_collate, pad_token_id=train_dataset.pad_token)

    model = create_model(
        vocab_size=train_dataset.vocab_size,
        pad_token_id=train_dataset.pad_token,
        beam_size=config.evaluation.beam_size,
        num_levels=train_dataset.num_levels,
        **config.model.model_dump(),
    ).to(device)
    total_params = get_num_params(model)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(
            inv_sqrt_schedule, constant_lr_steps=config.training.constant_lr_steps
        ),
    )

    sid_data = torch.load(
        config.data.semantic_ids_path,
        map_location="cpu",
        weights_only=False,
    )

    retrieval_metric_fn = partial(
        compute_metrics,
        num_levels=config.data.num_levels,
        beam_size=config.evaluation.beam_size,
        codebook_size=config.data.codebook_size,
        sid_to_asin=sid_data["sid_to_asin"],
        at_k=config.evaluation.at_k,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.evaluation.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        eval_strategy="steps",
        eval_steps=config.evaluation.every_steps,
        predict_with_generate=True,
        save_strategy="steps",
        save_steps=config.evaluation.every_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{config.evaluation.selection_metric}",
        greater_is_better=True,
        logging_steps=config.training.logging_steps,
        train_sampling_strategy="group_by_length",
        remove_unused_columns=False,
        dataloader_pin_memory=device.type == "cuda",
        seed=config.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=retrieval_metric_fn,
        optimizers=(optimizer, scheduler),
    )

    logger.info(
        f"Training: steps={config.training.max_steps:,}, train_samples={len(train_dataset):,}, "
        f"val_samples={len(val_dataset):,}, parameters={total_params:,}",
    )

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError(
            "No best checkpoint was selected. "
            "Check evaluation frequency and training length."
        )

    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))

    logger.info(
        f"Finished. Best {config.evaluation.selection_metric}={trainer.state.best_metric:.6f}; exported model: {best_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
