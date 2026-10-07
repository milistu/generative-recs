from pathlib import Path
from typing import Literal

import mmh3
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TigerDataset(Dataset):
    """
    Seq2seq dataset for TIGER generative retrival.

    Converts user interaction sequences into token sequences generative retrieval model.

    Input: [user_token, c1_0, c1_1, c1_2, c1_3, c2_0, ...]
    Target: [cn+1_0, cn+1_1, cn+1_2, cn+1_3]

    Args:
        splits_path: Path to splits.parquet
        semantic_ids_path: Path to semantic_ids.pt
        split: Split to load
        codebook_size: Number of tokens in the codebook
        num_levels: Number of levels in the RQ-VAE including collision token
        num_user_buckets: Number of user buckets (2000)
        max_seq_len: Maximum sequence length (20 items)
        seed: Random seed (42)
    """

    def __init__(
        self,
        splits_path: Path,
        semantic_ids_path: Path,
        split: Literal["train", "val", "test"] = "train",
        codebook_size: int = 256,
        num_levels: int = 4,
        num_user_buckets: int = 2000,
        max_seq_len: int = 20,
        sliding_window: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_levels = num_levels
        self.num_user_buckets = num_user_buckets
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Token offsets
        self.user_offset = codebook_size * num_levels
        self.pad_token = self.user_offset + num_user_buckets
        self.eos_token = self.pad_token + 1
        self.vocab_size = self.eos_token + 1

        # Max encoder length: 1 user token + max_seq_len * num_levels
        self.max_encoder_len = 1 + max_seq_len * num_levels

        # Load data
        df = pd.read_parquet(splits_path)
        sid_data = torch.load(semantic_ids_path, weights_only=False)
        self.asin_to_sid: dict[str, tuple[int, ...]] = sid_data["asin_to_sid"]

        # Build samples based on split
        self.samples = self._build_samples(df, split, sliding_window)

    def _hash_user(self, user_id: str) -> int:
        h = mmh3.hash(str(user_id), seed=self.seed, signed=False)
        return self.user_offset + (h % self.num_user_buckets)

    def _item_to_tokens(self, asin: str) -> list[int]:
        """Convert item ASIN to offset semantic ID tokens (one per level)."""
        sid = self.asin_to_sid[asin]
        return [level * self.codebook_size + code for level, code in enumerate(sid)]

    def _make_sample(
        self, user_token: int, history: list[str], target_asin: str
    ) -> dict[str, list[int]]:
        """Create a single (encoder_input, target) sample."""
        history_tokens = []
        for asin in history:
            history_tokens.extend(self._item_to_tokens(asin))

        return {
            "encoder_input": [user_token] + history_tokens,
            "target": self._item_to_tokens(target_asin),
        }

    def _build_samples(
        self,
        df: pd.DataFrame,
        split: Literal["train", "val", "test"],
        sliding_window: bool,
    ) -> list[dict[str, list[int]]]:
        """Build (encoder_input, target) pairs.

        For train split with sliding_window=True:
            User with train_items [a, b, c, d, e] creates:
            [a] -> b, [a,b] -> c, [a,b,c] -> d, [a,b,c,d] -> e

        For train split with sliding_window=False:
            Same user creates only: [a,b,c,d] -> e

        Val and test are always single-target (evaluation only).
        """
        samples = []
        for row in df.itertuples():
            user_token = self._hash_user(row.user_id)

            if split == "train":
                train_items = list(row.train_items)
                if sliding_window:
                    for i in range(1, len(train_items)):
                        history = train_items[:i][-self.max_seq_len :]
                        samples.append(
                            self._make_sample(user_token, history, train_items[i])
                        )
                else:
                    history = train_items[:-1][-self.max_seq_len :]
                    samples.append(
                        self._make_sample(user_token, history, train_items[-1])
                    )
            elif split == "val":
                history = list(row.train_items)[-self.max_seq_len :]
                samples.append(self._make_sample(user_token, history, row.val_item))
            else:
                history = (list(row.train_items) + [row.val_item])[-self.max_seq_len :]
                samples.append(self._make_sample(user_token, history, row.test_item))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_ids = torch.tensor(sample["encoder_input"], dtype=torch.long)
        labels = torch.tensor(sample["target"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


def custom_collate(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Pad histories to the longest history in the current batch."""
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )

    attention_mask = (input_ids != pad_token_id).long()
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
