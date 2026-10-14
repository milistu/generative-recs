# Generative Recs

Build, train, and evaluate generative recommender systems.
Starting with a reproduction of [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023). The method is referred to as **TIGER** (Transformer Index for GEnerative Recommenders).

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Download the dataset

This project uses the [Amazon Product Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) dataset (same as the TIGER paper). Download the **5-core** review data and metadata for your category of choice.

For example, for Toys and Games:

1. Download `reviews_Toys_and_Games_5.json.gz` (5-core reviews) and `meta_Toys_and_Games.json.gz` (metadata)
2. Place them in `data/2014/` and decompress:

```bash
gunzip data/2014/reviews_Toys_and_Games_5.json.gz
gunzip data/2014/meta_Toys_and_Games.json.gz
```

Your `data/2014/` directory should look like:

```
data/2014/
├── reviews_Toys_and_Games_5.json
└── meta_Toys_and_Games.json
```

For other categories (Beauty, Sports and Outdoors), find the corresponding files on the [dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and follow the same pattern.

## Usage

Run shell commands from the repository root. Configuration paths are relative to that directory.

### Prepare the data and Semantic IDs

To prepare the data and generate embeddings, run:

```bash
uv run --locked python scripts/prepare_data.py \
  --config configs/toys/preparation.yaml \
  --output-dir data/toys/processed
```

The command prints its output directory, such as `data/toys/processed/preparation_<timestamp>/`.

Before running `notebooks/02_train_rqvae.ipynb`, set its paths:

```python
DATA_DIR = Path("../data/toys/processed/<preparation_run>")
OUTPUT_DIR = Path("../checkpoints/toys/quantization/<quantization_run>")
```

Replace `<preparation_run>` with the directory created above and choose a new name for `<quantization_run>`. Notebook paths are relative to `notebooks/`.

Run the notebook to train the quantizer and generate Semantic IDs.

In the retrieval YAML you intend to use, update these fields:

```yaml
data:
  splits_path: data/toys/processed/<preparation_run>/splits.parquet
  semantic_ids_path: checkpoints/toys/quantization/<quantization_run>/semantic_ids.pt
```

Keep the other `data` settings unchanged. The Semantic IDs must come from the embeddings generated in the selected preparation run.

### Run a quick check

The smoke configuration runs 20 training steps on 1,024 training examples, with evaluation on 128 validation examples every 10 steps.

```bash
uv run python scripts/train.py \
  --config configs/toys/retrieval_smoke.yaml \
  --output-dir checkpoints/toys/retrieval
```

### Train the baseline

```bash
uv run python scripts/train.py \
  --config configs/toys/retrieval_baseline.yaml \
  --output-dir checkpoints/toys/retrieval
```

The baseline uses the full dataset and trains for 100,000 steps. By default, the checkpoint with highest validation Recall@10 is exported to `best/`.

`--output-dir` is the parent directory and can contain existing runs. Training creates a new `<config_name>_<UTC timestamp>/` subdirectory and logs its path. Settings are saved in `configs/retrieval.yaml` inside that run.

### Evaluate a saved model

Replace `<run_name>` with the run directory name printed by training.

Evaluate on validation data:

```bash
uv run python scripts/evaluate.py \
  --run-dir "checkpoints/toys/retrieval/<run_name>" \
  --split val
```

Evaluate it on test data:

```bash
uv run python scripts/evaluate.py \
  --run-dir "checkpoints/toys/retrieval/<run_name>" \
  --split test
```

Evaluation loads the run's saved `configs/retrieval.yaml` and `best/` model. Data paths in the configuration must point to existing files, relative to the repository root or as absolute paths. Evaluation reports Recall@K, NDCG@K, and loss.

Evaluation results are saved as `val_results.json` or `test_results.json` inside the run directory.

## Citations

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
