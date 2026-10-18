from math import isfinite
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from generative_recs.config import QuantizationTrainingConfig
from generative_recs.rqvae import RQVAE, UsageTracker


@torch.no_grad()
def evaluate_quantizer(
    model: RQVAE,
    embeddings: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    """Measure reconstruction loss and code usage across an item catalog.

    Evaluate after the quantizer's codebooks have been initialised.
    Recompute usage from scratch and restore the model's original mode.

    Usage is normalised entropy per quantizer level: higher values mean
    more evenly distributed assignments, not the fraction of active codes.

    Raises:
        ValueError: The embedding matrix contains no items.
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot evaluate an empty embedding matrix.")

    device = next(model.parameters()).device
    tracker = UsageTracker(
        num_quantizers=model.num_quantizers,
        codebook_size=model.codebook_size,
        device=device,
    )

    was_training = model.training
    total_recon_loss = 0.0

    model.eval()

    try:
        for batch in embeddings.split(batch_size):
            batch = batch.to(device)

            _, indices, recon_loss, _ = model(batch)

            total_recon_loss += recon_loss.item() * len(batch)
            tracker.update(indices)
    finally:
        model.train(was_training)

    return {
        "recon_loss": total_recon_loss / len(embeddings),
        **{f"usage_level_{level}": usage for level, usage in enumerate(tracker.get_usage())},
    }


def train_quantizer(
    model: RQVAE,
    embeddings: torch.Tensor,
    config: QuantizationTrainingConfig,
    run_dir: Path,
) -> None:
    """Train RQ-VAE, save catalog metrics, and reload the best checkpoint.

    The caller sets the random seed, places the model on its device, and
    creates run_dir. Embeddings contain the full item catalog.

    Select checkpoints by lowest catalog reconstruction loss. Stop after
    consecutive eligible evaluations without sufficient improvement.
    """
    device = next(model.parameters()).device
    loader = DataLoader(
        embeddings,
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = run_dir / "rqvae_best.pt"
    history = []
    step = 0
    best_loss = float("inf")
    best_step = 0
    stop_reference = float("inf")
    stale_evaluations = 0
    should_stop = False
    recon_sum = 0.0
    commit_sum = 0.0
    sample_count = 0

    model.train()
    while step < config.max_steps and not should_stop:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            _, _, recon_loss, commit_loss = model(batch)
            loss = recon_loss + commit_loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at step {step + 1}")

            loss.backward()
            optimizer.step()
            step += 1

            recon_sum += recon_loss.item() * len(batch)
            commit_sum += commit_loss.item() * len(batch)
            sample_count += len(batch)

            if step % config.evaluate_every != 0 and step != config.max_steps:
                continue

            metrics = evaluate_quantizer(model, embeddings, config.batch_size)
            current_loss = metrics["recon_loss"]
            usage = [metrics[f"usage_level_{i}"] for i in range(model.num_quantizers)]

            if not isfinite(current_loss):
                raise RuntimeError(f"Non-finite evaluation metric at step {step}")

            history.append(
                {
                    "step": step,
                    "train_recon_loss": recon_sum / sample_count,
                    "train_commit_loss": commit_sum / sample_count,
                    **{f"eval_{name}": value for name, value in metrics.items()},
                }
            )
            recon_sum = commit_sum = 0.0
            sample_count = 0

            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
                torch.save(model.state_dict(), checkpoint_path)

            pd.DataFrame(history).to_csv(
                run_dir / "training_history.csv",
                index=False,
            )
            logger.info(
                f"Step {step}: catalog reconstruction={current_loss:.6f}, "
                f"usage={[round(value, 3) for value in usage]}"
            )

            patience = config.early_stopping_patience
            eligible = (
                patience is not None
                and step >= config.early_stopping_min_steps
                and min(usage) >= config.early_stopping_usage
            )

            if not eligible:
                stop_reference = float("inf")
                stale_evaluations = 0
            elif current_loss < stop_reference - config.early_stopping_min_delta:
                stop_reference = current_loss
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                should_stop = stale_evaluations >= patience

            if should_stop:
                logger.info(f"Early stopping at step {step}.")

            if should_stop or step >= config.max_steps:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    logger.info(f"Loaded best checkpoint: step={best_step}, reconstruction={best_loss:.6f}")
