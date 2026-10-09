import numpy as np
from transformers import EvalPrediction


def _tokens_to_asin(
    tokens: list[int],
    sid_to_asin: dict[tuple[int, ...], str],
    codebook_size: int = 256,
    num_levels: int = 4,
) -> str | None:
    """Convert generated token IDs back to an ASIN via semantic ID lookup.

    The model generates offset tokens where level l uses range
    [l * codebook_size, (l + 1) * codebook_size - 1].

    Args:
        tokens: Generated token IDs (one per level, length == num_levels).
        sid_to_asin: Mapping from semantic ID tuple to ASIN.
        codebook_size: Number of codes per codebook level.
        num_levels: Number of levels in the semantic ID.

    Returns:
        ASIN string if the semantic ID is valid, None otherwise.
    """
    if len(tokens) != num_levels:
        return None

    codes = []
    for level, token in enumerate(tokens):
        expected_min = level * codebook_size
        expected_max = (level + 1) * codebook_size - 1
        if not (expected_min <= token <= expected_max):
            return None
        codes.append(token - expected_min)

    return sid_to_asin.get(tuple(codes))


def _check_metric_inputs(recommendations: list[list[str]], targets: list[str], k: int) -> None:
    """Validate inputs shared by the ranking metrics."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not targets:
        raise ValueError("targets must contain at least one user.")
    if len(recommendations) != len(targets):
        raise ValueError("recommendations and targets must have equal lengths.")


def recall_at_k(
    recommendations: list[list[str]],
    targets: list[str],
    k: int,
) -> float:
    """Return mean Recall@K for one target item per user.

    Each recommendation list contains unique items in ranked order.
    Each user scores 1 if their target appears in the first K items,
    otherwise 0. Empty recommendation lists score 0.

    Raises:
        ValueError: K is nonpositive, targets are empty, or user counts differ.
    """
    _check_metric_inputs(recommendations, targets, k)

    hits = sum(target in items[:k] for items, target in zip(recommendations, targets))
    return hits / len(targets)


def ndcg_at_k(
    recommendations: list[list[str]],
    targets: list[str],
    k: int,
) -> float:
    """Return mean NDCG@K for one target item per user.

    Each recommendation list contains unique items in ranked order.
    A target at rank r <= K scores 1 / log2(r + 1), with ranks starting
    at 1. Missing targets and empty recommendation lists score 0.

    Raises:
        ValueError: K is nonpositive, targets are empty, or user counts differ.
    """
    _check_metric_inputs(recommendations, targets, k)

    total = 0.0

    for items, target in zip(recommendations, targets):
        top_k = items[:k]
        if target in top_k:
            rank = top_k.index(target) + 1
            total += 1.0 / np.log2(rank + 1)

    return float(total / len(targets))


METRICS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
}


def compute_metrics(
    eval_pred: EvalPrediction,
    *,
    num_levels: int,
    beam_size: int,
    codebook_size: int,
    sid_to_asin: dict[tuple[int, ...], str],
    at_k: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Decode Seq2SeqTrainer generations and calculate ranking metrics.

    Predictions contain `beam_size` candidate rows per user, in beam order.
    Each row starts with START, followed by the Semantic ID.
    Labels contain one Semantic ID per user, without START.
    Trailing padding is ignored.

    Invalid candidates and duplicate items are removed, preserving order.
    Users without valid recommendations remain included in the scores.

    Returns:
        Mean metric values named like "recall@5" and "ndcg@5".

    Raises:
        ValueError: Configuration, prediction shape, or targets are invalid.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if min(num_levels, beam_size, codebook_size) <= 0:
        raise ValueError("ID dimensions and beam_size must be positive.")
    if not at_k or any(k <= 0 for k in at_k):
        raise ValueError("at_k must contain positive cutoffs.")

    for name, array in [("predictions", predictions), ("labels", labels)]:
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise ValueError(f"{name} must be a 2D NumPy array.")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must contain integer token IDs.")

    num_users = len(labels)

    if num_users == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if predictions.shape[0] != num_users * beam_size:
        raise ValueError("Expected beam_size generated sequences per user.")
    if predictions.shape[1] < 1 + num_levels or labels.shape[1] < num_levels:
        raise ValueError("Predictions or labels are shorter than the item ID.")

    # Remove START tokens and group each user's candidates together.
    candidates = predictions[:, 1 : 1 + num_levels].reshape(num_users, beam_size, num_levels)

    recommendations: list[list[str]] = []
    targets: list[str] = []

    for index, (user_candidates, label) in enumerate(zip(candidates, labels)):
        target = _tokens_to_asin(
            label[:num_levels].tolist(), sid_to_asin, codebook_size, num_levels
        )
        if target is None:
            raise ValueError(f"Unknown target Semantic ID at row {index}.")
        targets.append(target)

        items: list[str] = []
        seen: set[str] = set()

        for candidate in user_candidates:
            item = _tokens_to_asin(candidate.tolist(), sid_to_asin, codebook_size, num_levels)
            if item is not None and item not in seen:
                items.append(item)
                seen.add(item)

        recommendations.append(items)

    return {
        f"{name}@{k}": metric(recommendations, targets, k)
        for name, metric in METRICS.items()
        for k in at_k
    }
