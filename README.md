# Generative Recs

Build and experiment with generative recommender systems, starting with an implementation of [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
```

Download the **Toys and Games 5-core reviews and metadata** from [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
Place both archives in `data/2014/` and decompress them:

```bash
gunzip data/2014/reviews_Toys_and_Games_5.json.gz
gunzip data/2014/meta_Toys_and_Games.json.gz
```

## Run the pipeline

Run commands from the repository root. The supplied YAML configs connect the stages below.
Each preparation or training command writes directly to `--output-dir`, which must not already exist.

### 1. Prepare data

Create interaction splits, item text, and embeddings:

```bash
uv run --locked python scripts/prepare_data.py \
  --config configs/toys/preparation.yaml \
  --output-dir outputs/toys/baseline/preparation
```

### 2. Generate Semantic IDs

Train the quantizer and save its selected model and item-to-ID mappings:

```bash
uv run --locked python scripts/train_quantizer.py \
  --config configs/toys/quantization.yaml \
  --output-dir outputs/toys/baseline/quantization
```

### 3. Train the recommender

```bash
uv run --locked python scripts/train.py \
  --config configs/toys/retrieval_baseline.yaml \
  --output-dir outputs/toys/baseline/retrieval
```

The baseline trains for 100,000 steps and saves the model with the highest validation **Recall@10** in `retrieval/best/`.

For a quick check first, use the smoke config to run 20 steps on a subset of the data:

```bash
uv run --locked python scripts/train.py \
  --config configs/toys/retrieval_smoke.yaml \
  --output-dir outputs/toys/smoke/retrieval
```

### 4. Evaluate

```bash
uv run --locked python scripts/evaluate.py \
  --run-dir outputs/toys/baseline/retrieval \
  --split test
```

Use `--split val` for validation. Evaluation loads the saved model and config, then writes Recall@5/10, NDCG@5/10, and loss to `test_results.json` or `val_results.json` in the retrieval directory.

## Run another experiment

Copy the relevant YAML from `configs/toys/`, adjust its settings, and choose a descriptive output path such as `outputs/toys/lower_lr/retrieval`. Unchanged stages can reuse existing outputs.

For a new dataset or new Semantic IDs, update the input paths:

| Config | Input paths |
| --- | --- |
| Preparation | `reviews_path`, `metadata_path` |
| Quantization | `preparation_dir` |
| Retrieval | `data.splits_path`, `data.mappings_path` |

Train a new retrieval model when changing its Semantic IDs; existing model weights depend on the mappings used during training.

## Inspect the quantizer

Open [analyze_quantizer.ipynb](notebooks/analyze_quantizer.ipynb) and set `RUN_DIR` to a saved quantization directory. The notebook shows training curves, collisions, reconstruction quality, and similarity between items sharing ID prefixes.

## References

<details>
<summary>Paper and dataset citations</summary>

```bibtex
@article{rajput2023recommender,
  title={Recommender systems with generative retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and Hulikal Keshavan, Raghunandan and Vu, Trung and Heldt, Lukasz and Hong, Lichan and Tay, Yi and Tran, Vinh and Samost, Jonah and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={10299--10315},
  year={2023}
}
```

```bibtex
@inproceedings{he2016ups,
  title={Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering},
  author={He, Ruining and McAuley, Julian},
  booktitle={proceedings of the 25th international conference on world wide web},
  pages={507--517},
  year={2016}
}
```

```bibtex
@inproceedings{mcauley2015image,
  title={Image-based recommendations on styles and substitutes},
  author={McAuley, Julian and Targett, Christopher and Shi, Qinfeng and Van Den Hengel, Anton},
  booktitle={Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval},
  pages={43--52},
  year={2015}
}
```

</details>
