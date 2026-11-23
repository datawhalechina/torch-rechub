# Full-dataset benchmark results

These experiments used one RTX 5080 (16 GB), PyTorch 2.9.1+cu130 and the Torch-RecHub models at commit `c330c81b`. Each table reports held-out test scores using one validation-selected checkpoint per model. [Data preparation and run commands](../docs/en/tools/benchmark.md)

## MovieLens-1M

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

## Avazu_x1

All three models use the published Avazu_x1 splits and train for two epochs. Validation click AUC selects the checkpoint. Both metrics use predictions from that checkpoint, evaluated on the CPU across all 8,085,794 test impressions. Higher ROC AUC indicates better ordering of positives relative to negatives. Lower LogLoss indicates better agreement between predicted probabilities and observed labels. [FuxiCTR metric implementation](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/metrics.py)

| Model | Torch-RecHub AUC ↑ | Torch-RecHub LogLoss ↓ | BARS AUC ↑ | BARS LogLoss ↓ |
| --- | ---: | ---: | ---: | ---: |
| WideDeep | 0.758915 | 0.369409 | 0.7649 | 0.3665 |
| DeepFM | 0.758300 | 0.369325 | 0.7648 | 0.3667 |
| DCN | 0.759873 | 0.369670 | 0.7652 | 0.3665 |

The BARS settings use 10-dimensional embeddings on this split. These runs use 16 dimensions. The project example fits the vocabulary and scaler across all three splits and also casts the first three scaled fields to categorical IDs. Reproducing a published score requires matching the inputs and training settings as well as the metric. [BARS evaluation settings](https://openbenchmark.github.io/BARS/CTR/leaderboard/avazu_x1.html) · [BARS published scores](https://raw.githubusercontent.com/reczoo/RecBench/main/docs/CTR/leaderboard/avazu_x1.csv)

## Criteo

Both models trained for two epochs with seed 2022 on a random 70/10/20 split of impressions. ROC AUC and LogLoss use the same checkpoint evaluated on the CPU across all 9,168,125 test impressions. The scaler and vocabularies are fitted on training rows. DeepFM also maps categories occurring fewer than ten times in training to 0 and shares 26 categorical embeddings between its deep and FM branches.

| Model | AUC ↑ | LogLoss ↓ |
| --- | ---: | ---: |
| WideDeep | 0.790378 | 0.459636 |
| DeepFM | 0.804685 | 0.446696 |

## Census-Income

All models trained for ten epochs on the published splits and were evaluated on 49,881 test records. Validation income AUC selects a single checkpoint for all tasks. The targets are membership in the `50000+` income bracket and never having married. ESMM also predicts whether both conditions hold. The two PLE rows use different numbers of extraction levels.

| Model | Income AUC ↑ | Marital status AUC ↑ | Joint label AUC ↑ |
| --- | ---: | ---: | ---: |
| SharedBottom | 0.951358 | 0.994704 | — |
| ESMM | 0.841657 | 0.991467 | 0.970112 |
| MMOE | 0.951119 | 0.994779 | — |
| PLE (1 level) | 0.948204 | 0.994738 | — |
| PLE (2 levels) | 0.950635 | 0.994746 | — |
| AITM | 0.949838 | 0.994583 | — |

## AliExpress-US

Both models trained for two epochs. About 10% of search IDs from the training file were held out for validation. Validation conversion AUC selects the checkpoint. Both click and conversion AUC cover all 7,460,564 test impressions. The US archive is a public mirror. Two conversion labels differ in the 20 test rows checked against the bundled sample. The table uses the mirror labels. [US dataset mirror](https://drive.google.com/file/d/1oM9IcBYsaI7xJ2i6l-BnjUadU2YKquk0/view)

| Model | Click AUC ↑ | Conversion AUC ↑ |
| --- | ---: | ---: |
| SharedBottom | 0.698530 | 0.795083 |
| AITM | 0.692870 | 0.770054 |

## Data splits

| Dataset | Training examples | Validation examples | Test examples | Training batch size |
| --- | ---: | ---: | ---: | ---: |
| MovieLens-1M | 982,089 | 6,040 | 6,040 | 512 |
| Avazu_x1 | 28,300,276 | 4,042,897 | 8,085,794 | 8,192 |
| Criteo | 32,088,431 | 4,584,061 | 9,168,125 | 16,384 |
| Census-Income | 199,523 | 49,881 | 49,881 | 1,024 |
| AliExpress-US | 17,937,787 | 1,994,262 | 7,460,564 | 10,240 |
