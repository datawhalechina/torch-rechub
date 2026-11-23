---
title: Benchmarks
description: Compare recommendation models on the same data, reproduce full-dataset GPU runs, and generate a Markdown report.
---

# Benchmarks

Torch-RecHub models trained on one RTX 5080 (16 GB). Each table compares our models on the same data split and training budget. Retrieval metrics measure whether a recommendation list contains the user's next movie. The ranking and multi-task tables evaluate predictions of clicks, conversions or the specified labels. All scores reported below come from held-out test data.

## Results by dataset

### MovieLens-1M

All 1,000,209 ratings count as interactions. Each user's last movie is held out for testing and the preceding movie for validation. Each of the ten models trains for ten epochs. The checkpoint with the highest validation NDCG@10 then ranks the full catalog of 3,706 movies for 6,040 users.

| Model | GAUC | Recall@5 | Recall@10 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SINE | 0.837437 | 0.106623 | 0.168212 | 0.106623 | 0.168212 | 0.068642 | 0.088378 |
| STAMP | 0.839963 | 0.095199 | 0.149834 | 0.095199 | 0.149834 | 0.059889 | 0.077447 |
| GRU4Rec | 0.858918 | 0.090563 | 0.147351 | 0.090563 | 0.147351 | 0.056654 | 0.074937 |
| ComirecDR | 0.854196 | 0.082781 | 0.131126 | 0.082781 | 0.131126 | 0.052228 | 0.067701 |
| SASRec | 0.832628 | 0.080629 | 0.126159 | 0.080629 | 0.126159 | 0.051157 | 0.065777 |
| YoutubeDNN | 0.818883 | 0.058609 | 0.103477 | 0.058609 | 0.103477 | 0.035582 | 0.049994 |
| MIND | 0.827980 | 0.048510 | 0.084272 | 0.048510 | 0.084272 | 0.029637 | 0.041174 |
| DSSM | 0.805208 | 0.046854 | 0.083444 | 0.046854 | 0.083444 | 0.029233 | 0.040895 |
| ComirecSA | 0.834305 | 0.045861 | 0.081291 | 0.045861 | 0.081291 | 0.027935 | 0.039161 |
| NARM | 0.806611 | 0.039073 | 0.074503 | 0.039073 | 0.074503 | 0.023360 | 0.034827 |

Each user has one test target, so Recall@K equals Hit@K. NDCG also rewards placing the target nearer the top. GAUC averages AUC across users and gives tied positive–negative pairs half credit. [RecBole metric definitions](https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py)

All runs use seed 2022, the 20 most recent interactions and in-batch negatives. NARM, STAMP, SASRec and SINE use only the movie history. The other encoders also use the user ID. Evaluation excludes previously seen movies and retains the current target. MIND and both ComiRec variants use the highest score across four interests for each movie.

STAMP uses `emb_std=0.1` in this benchmark. With the example's 0.002 initialization, weight decay and in-batch training drove every movie's score to zero.

### Avazu_x1

All three models use the published Avazu_x1 splits and train for two epochs. Validation click AUC selects the checkpoint. Both metrics use predictions from that checkpoint, evaluated on the CPU across all 8,085,794 test impressions. Higher ROC AUC indicates better ordering of positives relative to negatives. Lower LogLoss indicates better agreement between predicted probabilities and observed labels. [FuxiCTR metric implementation](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/metrics.py)

| Model | Torch-RecHub AUC ↑ | Torch-RecHub LogLoss ↓ | BARS AUC ↑ | BARS LogLoss ↓ |
| --- | ---: | ---: | ---: | ---: |
| WideDeep | 0.758915 | 0.369409 | 0.7649 | 0.3665 |
| DeepFM | 0.758300 | 0.369325 | 0.7648 | 0.3667 |
| DCN | 0.759873 | 0.369670 | 0.7652 | 0.3665 |

The BARS settings use 10-dimensional embeddings on this split. These runs use 16 dimensions. The project example fits the vocabulary and scaler across all three splits and also casts the first three scaled fields to categorical IDs. Reproducing a published score requires matching the inputs and training settings as well as the metric. [BARS evaluation settings](https://openbenchmark.github.io/BARS/CTR/leaderboard/avazu_x1.html) · [BARS published scores](https://raw.githubusercontent.com/reczoo/RecBench/main/docs/CTR/leaderboard/avazu_x1.csv)

### Criteo

Both models trained for two epochs with seed 2022 on a random 70/10/20 split of impressions. ROC AUC and LogLoss use the same checkpoint evaluated on the CPU across all 9,168,125 test impressions. The scaler and vocabularies are fitted on training rows. DeepFM also maps categories occurring fewer than ten times in training to 0 and shares 26 categorical embeddings between its deep and FM branches.

| Model | AUC ↑ | LogLoss ↓ |
| --- | ---: | ---: |
| WideDeep | 0.790378 | 0.459636 |
| DeepFM | 0.804685 | 0.446696 |

### Census-Income

All models trained for ten epochs on the published splits and were evaluated on 49,881 test records. Validation income AUC selects a single checkpoint for all tasks. The targets are membership in the `50000+` income bracket and never having married. ESMM also predicts whether both conditions hold. The two PLE rows use different numbers of extraction levels.

| Model | Income AUC ↑ | Marital status AUC ↑ | Joint label AUC ↑ |
| --- | ---: | ---: | ---: |
| SharedBottom | 0.951358 | 0.994704 | — |
| ESMM | 0.841657 | 0.991467 | 0.970112 |
| MMOE | 0.951119 | 0.994779 | — |
| PLE (1 level) | 0.948204 | 0.994738 | — |
| PLE (2 levels) | 0.950635 | 0.994746 | — |
| AITM | 0.949838 | 0.994583 | — |

### AliExpress-US

Both models trained for two epochs. About 10% of search IDs from the training file were held out for validation. Validation conversion AUC selects the checkpoint. Both click and conversion AUC cover all 7,460,564 test impressions. The US archive is a public mirror. Two conversion labels differ in the 20 test rows checked against the bundled sample. The table uses the mirror labels. [US dataset mirror](https://drive.google.com/file/d/1oM9IcBYsaI7xJ2i6l-BnjUadU2YKquk0/view)

| Model | Click AUC ↑ | Conversion AUC ↑ |
| --- | ---: | ---: |
| SharedBottom | 0.698530 | 0.795083 |
| AITM | 0.692870 | 0.770054 |

## Compare models on your data

Run the commands from the repository root. Set up PyTorch for your device, then install the benchmark dependencies. Criteo and AliExpress need the `bigdata` extra to read data in chunks with PyArrow. For the other full datasets, install the dependencies with `uv sync --extra benchmark`. [Installation guide](../guide/install.md)

```bash
uv sync --extra benchmark --extra bigdata
```

Find the dataset in `benchmarks/configs/full.yaml`. Set `path` to the input file or directory and `models` to the models you want to train, in order. For a Criteo-format CSV of your own, include a header with the binary click label `label`, 13 numeric columns `I1` through `I13` and 26 categorical columns `C1` through `C26`.

```yaml
criteo:
  path: data/criteo/criteo.csv
  models: [WideDeep, DeepFM]
  epochs: 2
```

Activate the project environment and run the command below. Each model trains for two epochs, restores the weights with the highest validation AUC, and scores the same test set. Omit `epochs` to use each model's original training budget. When using your own sample, include both positive and negative examples for the primary task in the validation set so its AUC can select a checkpoint.

```bash
python -m benchmarks.full criteo --device cuda:0
```

Open `benchmark_results/full/criteo/suite.md` to compare the two models' test AUC and LogLoss. A dash means the metric does not apply or the test labels contain only one class, so AUC is undefined. The adjacent `suite.yaml` records the scores, data path, device and code revision. To change the input, model list or epoch count for one run, override them on the command line. Use `--output` to put the reports, weights and temporary encoded data on a drive with enough space.

```bash
python -m benchmarks.full census --data data/census-income --models MMOE PLE --epochs 10 --output benchmark_results/census_10
```

### Input files and supported models

| Dataset key | File or directory for `path` | Models |
| --- | --- | --- |
| `movielens` | `ml-1m.csv` built from the ratings, movies and users files. [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | YoutubeDNN, MIND, ComirecDR, ComirecSA, DSSM, GRU4Rec, NARM, STAMP, SASRec, SINE |
| `criteo` | Put all seven archive parts in one directory, extract from `.001` and use `criteo.csv`. [Criteo](https://aistudio.baidu.com/dataset/detail/389642) | WideDeep, DeepFM, DCN |
| `avazu` | Use the archive containing `train.csv`, `valid.csv` and `test.csv`. [Avazu_x1.zip](https://zenodo.org/records/5700987) | WideDeep, DeepFM, DCN |
| `amazon` | Full CSV with `user_id`, `item_id`, `cate_id` and `time`. [Amazon Electronics](https://aistudio.baidu.com/dataset/detail/355993) | DIN |
| `census` | Directory containing `census_income_{train,val,test}.csv`. [Census-Income](https://aistudio.baidu.com/dataset/detail/390912) | SharedBottom, ESMM, MMOE, PLE, PLE-2, AITM |
| `ali_ccp` | Directory containing `ali_ccp_{train,val,test}.csv`. [Ali-CCP](https://aistudio.baidu.com/datasetdetail/281072) | DeepFM, MMOE |
| `aliexpress` | US archive `AliExpress_US.zip` containing `train.csv` and `test.csv`. [Mirror used for these runs](https://drive.google.com/file/d/1oM9IcBYsaI7xJ2i6l-BnjUadU2YKquk0/view) | SharedBottom, AITM |

### Prepare the full MovieLens-1M dataset

Download `ml-1m.zip` from [GroupLens](https://grouplens.org/datasets/movielens/1m/). Place `ratings.dat`, `movies.dat` and `users.dat` in `examples/matching/data/ml-1m/`. With the project environment active, run the following from the repository root to build the full CSV and run the configured retrieval models.

```bash
cd examples/matching
python data/ml-1m/preprocess_ml.py
cd ../..
python -m benchmarks.full movielens --device cuda:0
```

## Run the bundled samples

The repository includes small datasets for the following tasks. You can try them on a CPU using the configurations in the corresponding `benchmarks/configs/` subdirectory.

| Subdirectory | Dataset and models | Metrics |
| --- | --- | --- |
| `matching/` | MovieLens · MIND, YoutubeDNN, ComirecDR, ComirecSA | Hit@K, Recall@K, NDCG@K, MRR@K, Precision@K |
| `ranking/` | Criteo · WideDeep, DeepFM, DCN | Choice of AUC, LogLoss, Accuracy and MSE |
| `multitask/` | Census-Income · ESMM, MMOE, PLE | AUC per task and mean AUC over tasks with a defined score |

The first command installs Annoy for retrieval. `runner.py` runs one configuration and writes `summary.md` and `result.yaml` to its `output_dir`. `suite.py` runs every matching configuration and writes `suite.md` and `suite.yaml` under `--output`.

```bash
uv sync --extra benchmark --extra annoy
python -m benchmarks.runner --config benchmarks/configs/matching/ml_1m_mind.yaml
python -m benchmarks.suite --configs "benchmarks/configs/matching/*.yaml" --output benchmark_results/suites/matching
```

After changing a model or trainer, add `--check-baseline` to the `runner.py` or `suite.py` command to check for regressions. The script reads reference values and tolerances from `benchmarks/baselines/<task>.yaml`. It exits with a nonzero status if a metric deteriorates beyond tolerance, for example if AUC drops or LogLoss increases.
