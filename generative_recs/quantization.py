from collections import Counter
from math import isfinite
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from generative_recs.config import QuantizationTrainingConfig
from generative_recs.rqvae import RQVAE, UsageTracker


@torch.no_grad()
def evaluate_quantizer(
    model: RQVAE,
    embeddings: torch.Tensor,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure reconstruction, code usage, and prefix collisions.

    Squared error is summed over embedding dimensions and averaged
    across items. Prefix collisions exclude the final collision counter.

    Evaluate after codebook initialization. Recompute catalog metrics
    from scratch and restore the model's original mode afterward.

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

    prefix_counts: Counter[tuple[int, ...]] = Counter()
    total_recon_loss = 0.0
    total_cosine_error = 0.0
    total_squared_error = 0.0
    was_training = model.training

    model.eval()

    try:
        for batch in embeddings.split(batch_size):
            batch = batch.to(device)

            reconstructed, indices, recon_loss, _ = model(batch)

            total_recon_loss += recon_loss.item() * len(batch)
            total_cosine_error += (
                (1 - F.cosine_similarity(batch, reconstructed, dim=1)).sum().item()
            )
            total_squared_error += (batch - reconstructed).square().sum(dim=1).sum().item()
            tracker.update(indices)
            prefix_counts.update(tuple(codes) for codes in indices.cpu().tolist())
    finally:
        model.train(was_training)

    metrics: dict[str, float | int] = {
        "num_items": len(embeddings),
        "recon_loss": total_recon_loss / len(embeddings),
        "cosine_error": total_cosine_error / len(embeddings),
        "squared_l2_error": total_squared_error / len(embeddings),
        "unique_prefixes": len(prefix_counts),
        "collision_groups": sum(count > 1 for count in prefix_counts.values()),
        "items_in_collision_groups": sum(count for count in prefix_counts.values() if count > 1),
        "largest_collision_group": max(prefix_counts.values()),
    }
    for level, entropy in enumerate(tracker.get_normalized_entropy()):
        metrics[f"normalized_entropy_level_{level}"] = entropy
    for level, fraction in enumerate(tracker.get_active_fraction()):
        metrics[f"active_fraction_level_{level}"] = fraction
    return metrics


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
            entropy = [
                metrics[f"normalized_entropy_level_{i}"] for i in range(model.num_quantizers)
            ]

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
                f"entropy={[round(value, 3) for value in entropy]}"
            )

            patience = config.early_stopping_patience
            eligible = (
                patience is not None
                and step >= config.early_stopping_min_steps
                and min(entropy) >= config.early_stopping_usage
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


@torch.no_grad()
def generate_semantic_ids(
    model: RQVAE,
    asins: list[str],
    embeddings: torch.Tensor,
    batch_size: int,
) -> dict[str, tuple[int, ...]]:
    """Generate unique item IDs using a trained quantizer.

    ASINs and embedding rows must have the same order. Append a collision
    counter to each quantized ID, assigning counters in the supplied order.
    Restore the model's original training/evaluation mode afterward.

    Raises:
        ValueError: ASINs and embeddings differ in length, ASINs repeat,
            or a collision group exceeds the available counter tokens.
    """
    if len(asins) != len(embeddings):
        raise ValueError("ASINs and embeddings must have the same length.")
    if len(set(asins)) != len(asins):
        raise ValueError("ASINs must be unique.")

    device = next(model.parameters()).device
    counts: Counter[tuple[int, ...]] = Counter()
    asin_to_sid: dict[str, tuple[int, ...]] = {}
    was_training = model.training

    model.eval()
    try:
        for start in range(0, len(asins), batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            indices = model.get_semantic_ids(batch).cpu().tolist()
            batch_asins = asins[start : start + batch_size]

            for asin, codes in zip(batch_asins, indices, strict=True):
                prefix = tuple(codes)
                counter = counts[prefix]

                if counter >= model.codebook_size:
                    raise ValueError(
                        f"More than {model.codebook_size} items share ID {prefix}. "
                        f"The collision counter exceeds the retrieval token range."
                    )

                sid = prefix + (counter,)
                asin_to_sid[asin] = sid
                counts[prefix] += 1

    finally:
        model.train(was_training)

    logger.info(
        f"Generated IDs for {len(asins)} items: {len(counts)} unique prefixes, "
        f"{sum(count > 1 for count in counts.values())} collision groups, "
        f"largest group={max(counts.values(), default=0)}"
    )

    return asin_to_sid
