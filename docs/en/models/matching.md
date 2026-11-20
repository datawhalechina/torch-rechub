---
title: Retrieval models
description: Retrieval models, training examples and Top-K evaluation in Torch-RecHub
---

# Retrieval models

In a movie recommender, retrieval selects an initial set of candidates from the whole catalog. With DSSM, you can compute all movie vectors after training. When a user arrives, compute one user vector and search for nearby movies. The [MovieLens retrieval tutorial](/tutorials/retrieval) shows how to map search results back to movie IDs.

## 1. DSSM

DSSM (Deep Structured Semantic Models) has two towers. The user tower reads user attributes or watch history. The item tower reads attributes such as the movie ID. Both towers output vectors of the same dimension.

### Paper

```
Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
```

This implementation L2-normalizes both vectors. By default, it takes the dot product of the user and movie vectors from the same row and applies a sigmoid. The loss compares this score with the row’s binary label. With in-batch negatives, the trainer also takes movies from other rows in the batch as candidates.

### Usage

```python
import os

from torch_rechub.models.matching import DSSM
from torch_rechub.basic.features import SparseFeature, DenseFeature

# Define the features
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# Create the model
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # Similarity scaling for in-batch training
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User features | Required |
| item_features | list | Item features | Required |
| temperature | float | Used for in-batch scoring. DSSM’s default `forward` does not scale the score. | 1.0 |
| user_params | dict | User tower network parameters | Required |
| item_params | dict | Item tower network parameters | Required |

## 2. FaceBookDSSM

FaceBookDSSM compares two movies for one user in each training example. One movie received positive feedback from the user. The other is a sampled negative.

The two movies share the item tower. The default forward pass returns their positive and negative scores. `MatchTrainer(mode=1)` uses BPR loss to push the positive score above the negative score.

### Usage

```python
from torch_rechub.models.matching import FaceBookDSSM

# Create the model
model = FaceBookDSSM(
    user_features=user_features,
    pos_item_features=item_features,
    neg_item_features=neg_item_features,
    user_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
    item_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User features | Required |
| pos_item_features | list | Positive item features | Required |
| neg_item_features | list | Negative item features corresponding to the positive item | Required |
| user_params | dict | User tower network parameters | Required |
| item_params | dict | Parameters of the item tower shared by positive and negative items | Required |

## 3. YoutubeDNN

YoutubeDNN predicts the next movie from user features and watch history. The user tower computes a vector with an MLP. On the item side, the model looks up the movie ID in an embedding table.

### Paper

```
Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
```

The example first averages the movie vectors in the history. During training, the target movie goes first in the candidate list, followed by sampled negatives. The user and movie vectors need the same width for the dot product.

### Usage

```python
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import YoutubeDNN

user_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=16),
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="mean", shared_with="item_id", padding_idx=0),
]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16, padding_idx=0)]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
model = YoutubeDNN(
    user_features=user_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User features | Required |
| item_features | list | Item features | Required |
| neg_item_feature | list | Sequence of negative IDs. Use `pooling="concat"` and share the positive ID embedding. | Required |
| user_params | dict | User tower network parameters | Required |
| temperature | float | Temperature | 1.0 |

## 4. YoutubeSBC

YoutubeSBC accounts for item sampling probabilities when scoring user–movie pairs. Popular movies appear more often in a batch, so the candidates in a batch have a different distribution from uniform catalog samples.

### Paper

```
Wu, Liang, et al. "RecSys 2019 tutorial: Deep learning for recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

The model computes cosine similarities, then subtracts the log of each item’s sampling weight from its column of scores. Prepare the `sample_weight_feature` before training.

### Usage

```python
from torch_rechub.models.matching import YoutubeSBC

# Create the model
model = YoutubeSBC(
    user_features=user_features,
    item_features=item_features,
    sample_weight_feature=sample_weight_feature,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    batch_size=256,
    n_neg=3,
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User features | Required |
| item_features | list | Item features | Required |
| sample_weight_feature | list | Item sampling weights used for sampling-bias correction | Required |
| user_params | dict | User tower network parameters | Required |
| item_params | dict | Item tower network parameters | Required |
| batch_size | int | Must match the DataLoader batch size | Required |
| n_neg | int | In-batch negatives per positive. Must be smaller than the batch size. | 3 |
| temperature | float | Temperature | 1.0 |

## 5. MIND

MIND keeps multiple interest vectors per user. If a user watches both animation and thrillers, each vector can retrieve candidates independently. The results are then merged into one list.

### Paper

```
Chen, Jiaxi, et al. "Multi-interest network with dynamic routing for recommendation at Tmall." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

Dynamic routing aggregates the watch history into several interests. During training the target item helps select or combine relevant interests. At inference the user keeps multiple interest vectors.

### Usage

```python
from torch_rechub.models.matching import MIND

user_features = [SparseFeature("user_id", vocab_size=10000, embed_dim=16)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16)]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]

# Create the model
model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequence user features | Required |
| history_features | list | Historical item sequence with `pooling="concat"` | Required |
| item_features | list | Positive item ID features | Required |
| neg_item_feature | list | Negative ID sequence with `pooling="concat"` | Required |
| max_length | int | Fixed length of the input history | Required |
| interest_num | int | Number of interests to learn | 4 |
| temperature | float | Temperature | 1.0 |

## 6. GRU4Rec

GRU4Rec reads the viewing history in order. The sequence of earlier movies and the most recent movie both affect the final user vector.

### Paper

```
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
```

This implementation concatenates the final GRU hidden state with user attributes and sends the result through an MLP. The MovieLens example puts the positive movie in the first column. The trainer computes cross-entropy over that movie and the presampled negatives.

### Usage

```python
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import GRU4Rec

user_features = [SparseFeature("user_id", vocab_size=10000, embed_dim=16)]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User features | Required |
| history_features | list | Historical sequence with `pooling="concat"`. This implementation uses only the first sequence. | Required |
| item_features | list | Positive item ID features | Required |
| neg_item_feature | list | Negative ID sequence with `pooling="concat"` | Required |
| user_params | dict | User MLP parameters. Its last layer must match the item embedding dimension. | Required |
| temperature | float | Temperature | 1.0 |

GRU4Rec takes the dot product over the embedding dimension for each user and candidate. Each output row contains the scores of the positive and negative items.

## 7. NARM

NARM combines two descriptions of a session. The final GRU state summarizes the whole history. Attention selects information from individual states that is relevant to the current session.

### Paper

```
Li, Jing, et al. "Neural attentive session-based recommendation." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
```

The model concatenates both parts to form a session vector, then projects it to the item embedding dimension to score candidates.

### Usage

```python
from torch_rechub.models.matching import NARM

# Create the model
model = NARM(
    item_history_feature=item_history_feature,
    hidden_dim=64,
    emb_dropout_p=0.2,
    session_rep_dropout_p=0.2,
    item_feature=item_feature,  # Needed only for in-batch negatives or standalone item-tower inference
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | Item sequence for a session. Zero is padding. | Required |
| hidden_dim | int | GRU hidden dimension | Required |
| emb_dropout_p | float | Item embedding dropout | Required |
| session_rep_dropout_p | float | Session representation dropout | Required |
| item_feature | SparseFeature or None | Optional target item ID for the standalone item tower or in-batch negatives | None |

## 8. SASRec

SASRec applies self-attention to item IDs in chronological order. Each position can see its current item and the preceding interactions to predict the next item.

### Paper

```
Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.
```

Positional embeddings distinguish earlier and later interactions. Training uses a history with matching positive and negative target sequences. At inference the history produces a user vector.

### Usage

```python
from torch_rechub.models.matching import SASRec

# Create the model
model = SASRec(
    features=[seq_feature, pos_feature, neg_feature],
    max_len=50,
    dropout_rate=0.2,
    num_blocks=2,
    num_heads=4,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Three `SequenceFeature` objects, ordered as history, positive targets and negative targets | Required |
| max_len | int | Sequence length | 50 |
| dropout_rate | float | Dropout for embeddings, attention and the feed-forward network | 0.5 |
| num_blocks | int | Self-attention blocks | 2 |
| num_heads | int | Number of attention heads. Must divide the embedding dimension. | 1 |
| item_feature | SparseFeature or None | Experimental standalone item tower feature | None |

SASRec builds the mask and position indices on the same device as the sequence. Configure `item_feature` to extract target item vectors independently.

## 9. SINE

SINE selects a small subset of interest prototypes for each user from a pool shared across all users. “Sparse” refers to how few prototypes it selects.

### Paper

```
Chen, Jiaxi, et al. "SINE: A sparse interest network for sequential recommendation." Proceedings of the 14th ACM Conference on Recommender Systems. 2021.
```

`num_concept` sets the total number of prototypes. `num_intention` sets how many to select per user. The model matches the history to the prototypes, selects relevant ones and aggregates them into one user vector.

### Usage

```python
from torch_rechub.models.matching import SINE

# Create the model
model = SINE(
    history_features=["hist_item_id"],
    item_features=["item_id"],
    neg_item_features=["neg_items"],
    num_items=100000,
    embedding_dim=32,
    hidden_dim=64,
    num_concept=100,
    num_intention=4,
    seq_max_len=50,
    num_heads=4,
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| history_features | list[str] | History sequence field names. Only the first is used. | Required |
| item_features | list[str] | Positive item field names | Required |
| neg_item_features | list[str] | Negative item sequence field names | Required |
| num_items | int | Item vocabulary size including padding | Required |
| embedding_dim | int | Embedding dimension | Required |
| hidden_dim | int | Attention hidden dimension | Required |
| num_concept | int | Number of global concept prototypes | Required |
| num_intention | int | Number of intentions selected per user | Required |
| seq_max_len | int | Fixed input sequence length | Required |
| num_heads | int | Self-attention heads | 1 |
| temperature | float | Temperature | 1.0 |

## 10. STAMP

STAMP predicts the next item from the clicks in a session. It also singles out the last clicked item to let the most recent choice influence the score directly.

### Paper

```
Liu, Qiao, et al. "STAMP: short-term attention/memory priority model for session-based recommendation." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

The model combines information from the full history and the last click into a session vector, then scores it against the item vectors.

### Usage

```python
from torch_rechub.models.matching import STAMP

# Create the model
model = STAMP(
    item_history_feature=item_history_feature,
    weight_std=0.05,
    emb_std=0.05,
    item_feature=item_feature,  # Needed only for in-batch negatives or standalone item-tower inference
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | Item sequence for a session. Zero is padding. | Required |
| weight_std | float | Standard deviation for attention parameter initialization | Required |
| emb_std | float | Standard deviation for embedding and linear-layer initialization | Required |
| item_feature | SparseFeature or None | Optional target item ID for the item tower or in-batch negatives | None |

## 11. ComirecDR

ComirecDR uses dynamic routing to assign interactions to different interests. This lets a user keep multiple vectors for different preferences.

### Paper

```
Chen, Jiaxi, et al. "Controllable multi-interest framework for recommendation." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
```

All interests search the same set of item vectors. More interest vectors mean more searches per user.

### Usage

```python
from torch_rechub.models.matching import ComirecDR

# Create the model
model = ComirecDR(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequence user features | Required |
| history_features | list | Historical item sequence with `pooling="concat"` | Required |
| item_features | list | Positive item features | Required |
| neg_item_feature | list | Negative item sequence with `pooling="concat"` | Required |
| max_length | int | Fixed history length | Required |
| interest_num | int | Number of interests | 4 |
| temperature | float | Temperature | 1.0 |

## 12. ComirecSA

ComirecSA also keeps multiple interest vectors. The SA variant summarizes the history with several sets of attention weights, one set for each interest.

For example, two sets of weights may emphasize different movies in the history. Whether they learn distinct interests depends on the training data.

### Usage

```python
from torch_rechub.models.matching import ComirecSA

# Create the model
model = ComirecSA(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    interest_num=4,
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequence user features | Required |
| history_features | list | Historical item sequence with `pooling="concat"` | Required |
| item_features | list | Positive item features | Required |
| neg_item_feature | list | Negative item sequence with `pooling="concat"` | Required |
| interest_num | int | Number of interests | 4 |
| temperature | float | Temperature | 1.0 |

## 13. Choosing a retrieval model

| Training examples or objective | Start with | What to look for in the code |
| --- | --- | --- |
| A user, a movie and a binary label per row | DSSM | How user fields and item fields are assigned to separate towers |
| A positive and a negative movie for the same user | FaceBookDSSM, YoutubeDNN | The former compares two scores. The latter identifies the positive in a candidate list. |
| Clicks in chronological order | GRU4Rec, NARM, STAMP, SASRec | How order and the last click contribute to the prediction |
| Retrieve several sets of candidates from the same history | MIND, ComirecDR, ComirecSA | How many vectors each user produces and how their results are merged |

## 14. From training to retrieval

In the [retrieval tutorial](/tutorials/retrieval), both user and item fields use 16-dimensional sparse embeddings. The final layer of each tower produces a 64-dimensional vector. Use the same MovieLens data to compare the Top-K results for default DSSM training and in-batch negatives.

## 15. Reading Top-K results

Suppose a user has two held-out movies and the first ten candidates include one of them. Recall@10 is 1/2 because one of the two relevant movies was retrieved. Precision@10 is 1/10 because one of ten candidates is relevant. Hit@10 is 1 as soon as any relevant movie appears. To distinguish a hit at rank one from a hit at rank ten, use NDCG@10.

## 16. Saving the item index

Build the index one row per movie vector and save the original movie IDs in exactly that order. For each row number returned by the index, look up the movie in that list. After updating the model weights, recompute the movie vectors and rebuild the index. [Vector indexing](/serving/vector_index)
