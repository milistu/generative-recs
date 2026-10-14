import ast
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from generative_recs.config import EmbeddingConfig


def load_reviews(path: Path) -> pd.DataFrame:
    """Load Amazon 2014 JSON Lines reviews with standard column names.

    Return user_id, asin, rating, and timestamp in source-file order.
    """
    reviews = pd.read_json(
        path,
        lines=True,
        dtype={"reviewerID": str, "asin": str},
    )

    return reviews.rename(
        columns={
            "reviewerID": "user_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
        }
    )[["user_id", "asin", "rating", "timestamp"]].copy()


def build_splits(reviews: pd.DataFrame) -> pd.DataFrame:
    """Reserve each user's last two interactions for validation and test.

    Sort interactions by timestamp, preserving input row order for ties.
    Keep the complete remaining history for training.

    Raises:
        ValueError: A user has fewer than three interactions.
    """
    ordered_reviews = reviews.assign(source_order=range(len(reviews))).sort_values(
        ["user_id", "timestamp", "source_order"]
    )

    sequences = ordered_reviews.groupby("user_id", sort=True)["asin"].agg(list)

    if (sequences.map(len) < 3).any():
        raise ValueError(
            "Each user needs at least three interactions for train/val/test splitting."
        )

    return pd.DataFrame(
        [
            {
                "user_id": user_id,
                "train_items": items[:-2],
                "val_item": items[-2],
                "test_item": items[-1],
            }
            for user_id, items in sequences.items()
        ]
    )


def format_item_text(item: dict[str, Any]) -> str:
    """Combine Amazon item metadata into text for embedding.

    Place title, brand, categories, and price before the description.
    Omit missing fields.
    """
    parts = []

    if item.get("title"):
        parts.append(str(item["title"]))

    if item.get("brand"):
        parts.append(f"Brand: {item['brand']}")

    categories = item.get("categories")
    if categories:
        category_paths = [" > ".join(path) for path in categories if path]
        if category_paths:
            parts.append("Category: " + "; ".join(category_paths))

    price = item.get("price")
    if price is not None:
        parts.append(f"Price: ${price}")

    if item.get("description"):
        parts.append(str(item["description"]))

    return "\n".join(parts) if parts else "Unknown product"


def load_items(path: Path, item_ids: set[str]) -> pd.DataFrame:
    """Load metadata and build embedding text for the requested items.

    Return one row per item with asin and text columns, in source-file order.

    Raises:
        ValueError: Requested items have missing or duplicate metadata.
    """
    items = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = ast.literal_eval(line)

            if record.get("asin") in item_ids:
                items.append(
                    {
                        "asin": record["asin"],
                        "text": format_item_text(record),
                    }
                )

    result = pd.DataFrame(items, columns=["asin", "text"])

    missing = item_ids - set(result["asin"])
    if missing:
        raise ValueError(f"Metadata is missing for {len(missing)} requested items.")

    if result["asin"].duplicated().any():
        raise ValueError("Metadata contains duplicate item IDs.")

    return result


def generate_embeddings(
    items: pd.DataFrame,
    config: EmbeddingConfig,
) -> dict[str, torch.Tensor]:
    """Encode item text and return CPU embeddings keyed by ASIN.

    Report text lengths before truncation, including special tokens.
    Use the configured sequence limit and batch size for encoding.
    """
    model = SentenceTransformer(config.model_name)
    model.max_seq_length = config.max_seq_length

    texts = items["text"].tolist()
    asins = items["asin"].tolist()

    longest = 0
    truncated_count = 0

    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        tokenized = model.tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            return_attention_mask=False,
            verbose=False,
        )
        lengths = tokenized["length"]
        longest = max(longest, max(lengths))
        truncated_count += sum(length > config.max_seq_length for length in lengths)

    logger.info(
        f"Embedding {len(texts)} items: token limit={config.max_seq_length}, "
        f"longest text={longest}, texts over limit={truncated_count}"
    )

    if truncated_count:
        logger.warning(
            f"{truncated_count} item texts will be truncated to {config.max_seq_length} tokens."
        )

    embeddings = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).cpu()

    return dict(zip(asins, embeddings, strict=True))
