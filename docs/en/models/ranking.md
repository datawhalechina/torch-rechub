---
title: Ranking models
description: Ranking models, feature interactions and in-batch negatives in Torch-RecHub
---

# Ranking models

A ranking model scores each user–candidate pair. For example, to rank ten movies for a user’s home page, construct ten rows with the same user features and different movie features. Sort the movies by the ten predicted scores. Start with the [DeepFM quick start](/guide/quick_start) for training code. The examples below show how to construct the individual models.

## In-batch negatives with CTRTrainer

When your data contains impressions and click labels, the default training path uses those labels directly. To construct negatives from items clicked by other users in the same batch, tell `CTRTrainer` which columns belong to the item.

```python
trainer = CTRTrainer(model, in_batch_neg=True,
                     item_features=["item_id", "cate_id"])
```

Suppose Alex clicks science fiction movie A and Sam clicks comedy B. The trainer selects rows with label 1. It keeps Alex’s user features, history and context, takes both `item_id` and `cate_id` from B, and gives the new row label 0. Replacing only the movie ID would pair comedy B with A’s science fiction category.

`item_features` accepts column names or existing Feature objects. Include the item ID and every attribute that changes with it. A cross feature such as “how many movies of this genre the user watched in the last seven days” needs to be recomputed when the movie changes. The trainer does not do that automatically. For anonymous Criteo fields, you need to know which fields describe the item before enabling this option.

- By default, each positive is paired with a randomly selected item from another positive row. A row cannot select itself, but two rows can contain the same item. Set `in_batch_neg_ratio` to a positive integer to sample that many items with replacement or to `None` to use all other positive rows.
- `sampler_seed` controls the random sampling. A batch with fewer than two positive rows is skipped. If there are no usable pairs over an entire epoch, the trainer raises an error.
- The Amazon DIN and DIEN examples already define the item columns. Add `--in_batch_neg` to switch modes. DIEN still needs negative history sequences for its auxiliary loss. Other single-task ranking examples also require `--item_features`.

A synthetic negative is a pair assigned label 0 for training. Evaluate click prediction on held-out impressions with their observed click labels. To prepare a training set containing only positives, use `generate_seq_feature(..., in_batch_neg=True)`. The validation and test sets keep their original labels.

## 1. WideDeep

WideDeep sends each impression through two branches. The wide branch scores selected fields with a linear layer. The deep branch concatenates its fields and passes them through an MLP. The two scores are added, then a sigmoid produces the click score.

### Paper

```
Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
```

`wide_features` and `deep_features` determine which fields go into each branch.

### Usage

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import WideDeep

# Define the features
dense_features = [DenseFeature(name="age", embed_dim=1), DenseFeature(name="income", embed_dim=1)]
sparse_features = [SparseFeature(name="city", vocab_size=100, embed_dim=16), SparseFeature(name="gender", vocab_size=3, embed_dim=16)]

# Create the model
model = WideDeep(
    wide_features=sparse_features,
    deep_features=sparse_features + dense_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| wide_features | list | Fields used by the wide branch | Required |
| deep_features | list | Fields used by the deep branch | Required |
| mlp_params | dict | Deep network parameters, including dims, dropout and activation | Required |

## 2. DeepFM

DeepFM uses an FM branch for pairwise field interactions and an MLP for concatenated features. For example, the model can assign different scores to different movie genres even when both impressions occur on the weekend.

### Paper

```
Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." Proceedings of the 26th international joint conference on artificial intelligence. 2017.
```

The FM branch stacks fields to take dot products, so the sparse features in `fm_features` need a common embedding width. When a field appears in both branches, they share its embedding.

### Usage

```python
from torch_rechub.models.ranking import DeepFM

# Create the model
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Fields used by the deep branch | Required |
| fm_features | list | Features used by the FM branch | Required |
| mlp_params | dict | Deep network parameters | Required |

## 3. DCN

DCN (Deep & Cross Network) reintroduces the original features at each cross layer. Deeper layers can combine the interactions learned by earlier ones. The cross and deep branch outputs are used together to predict clicks.

### Paper

```
Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017.
```

### How it works

- **Cross network.** Each layer explicitly learns feature crosses of increasing order. Its output is
  $$x_{l+1} = x_0 x_l^T w_l + b_l + x_l$$
- **Deep network.** An MLP learns nonlinear feature interactions.
- **Joint training.** The cross and deep networks run in parallel. Their outputs are concatenated and passed to a fully connected layer.

### Usage

```python
from torch_rechub.models.ranking import DCN

# Create the model
model = DCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features used by both the cross and deep branches | Required |
| mlp_params | dict | Deep network parameters | Required |
| n_cross_layers | int | Number of cross layers | Required |

## 4. DCNv2

DCNv2 replaces the weight vector in a DCN cross layer with a matrix. This gives individual pairs of dimensions separate interaction weights and increases the parameter count.

### Paper

```
Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021.
```

`use_low_rank_mixture=True` uses low-rank experts to reduce the cost of large matrices. `model_structure` selects a cross-only network or a serial or parallel combination of cross and deep networks.

### Usage

```python
from torch_rechub.models.ranking import DCNv2

# Create the model
model = DCNv2(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    model_structure="parallel",       # crossnet_only / stacked / parallel
    use_low_rank_mixture=True,
    low_rank=32,
    num_experts=4,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features used by both the cross and deep branches | Required |
| mlp_params | dict | Deep network parameters | Required |
| n_cross_layers | int | Number of cross layers | Required |
| model_structure | str | `crossnet_only` / `stacked` / `parallel` | `parallel` |
| use_low_rank_mixture | bool | Use a mixture of low-rank cross-network experts | True |
| low_rank | int | Low-rank dimension | 32 |
| num_experts | int | Number of CrossNetMix experts | 4 |

## 5. EDCN

EDCN adds a bridge between the cross network and the deep network. At each layer the two branches can exchange features. The final score uses both branches and the bridge output.

### Paper

```
Ma, Xiao, et al. "Enhanced Deep & Cross Network for Feature Cross Learning in Click-Through Rate Prediction." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

`bridge_type` selects how to combine the branch outputs. The regulation module learns separate field weights for each branch.

### Usage

```python
from torch_rechub.models.ranking import EDCN

# Create the model
model = EDCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    bridge_type="hadamard_product",
    use_regulation_module=True,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | All input features | Required |
| mlp_params | dict | Deep network parameters | Required |
| n_cross_layers | int | Number of cross layers | Required |
| bridge_type | str | How to combine the cross and deep branches | `hadamard_product` |
| use_regulation_module | bool | Enable the feature regulation module | True |

## 6. AFM

AFM learns a weight for each pairwise field interaction. For instance, time of day × movie genre and city × time of day can receive different weights.

### Paper

```
Xiao, Jun, et al. "Attentional factorization machines: Learning the weight of feature interactions via attention networks." arXiv preprint arXiv:1708.04617 (2017).
```

It multiplies each pair of field embeddings element by element, then computes an attention-weighted sum of the resulting interaction vectors.

### Usage

```python
from torch_rechub.models.ranking import AFM

# Create the model
model = AFM(
    fm_features=sparse_features,
    embed_dim=16,  # Must match the embedding width of fm_features
    t=64,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| fm_features | list | Sparse features used in FM interactions. Use one embedding width for all of them. | Required |
| embed_dim | int | Feature embedding width | Required |
| t | int | Attention hidden dimension | 64 |

## 7. FiBiNET

FiBiNET adjusts the weight of each field for the current example, then computes pairwise interactions with bilinear transformations. A field’s contribution can depend on the field it is paired with.

### Paper

```
Juan, Yuchin, et al. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

The model computes bilinear interactions for both the original and reweighted embeddings and uses both sets to score the example.

### Usage

```python
from torch_rechub.models.ranking import FiBiNet

# Create the model
model = FiBiNet(
    features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    reduction_ratio=3,
    bilinear_type="field_interaction",
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features used by SENET and the bilinear interaction layer | Required |
| mlp_params | dict | Prediction MLP parameters | Required |
| reduction_ratio | int | SENET reduction ratio | 3 |
| bilinear_type | str | `field_all` / `field_each` / `field_interaction` | `field_interaction` |

## 8. DeepFFM

DeepFFM takes field identity into account. A city’s vector when interacting with age can differ from its vector when interacting with movie genre.

### Paper

```
Xiao, Jun, et al. "Deep learning over multi-field categorical data." European conference on information retrieval. Springer, Cham, 2016.
```

With many fields, check how many embedding parameters this requires. The example also shows how to construct `FatDeepFFM` with field attention.

### Usage

```python
from torch_rechub.models.ranking import DeepFFM, FatDeepFFM

# FFM uses different embeddings per field pair. Scale the cross-feature vocabulary by the number of fields.
ffm_linear_features = [
    SparseFeature(f.name, vocab_size=f.vocab_size, embed_dim=1) for f in sparse_features
]
ffm_cross_features = [
    SparseFeature(f.name, vocab_size=f.vocab_size * len(sparse_features), embed_dim=10)
    for f in sparse_features
]

model = DeepFFM(
    linear_features=ffm_linear_features,
    cross_features=ffm_cross_features,
    embed_dim=10,
    mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"},
)

# Create the FatDeepFFM variant
fat_model = FatDeepFFM(
    linear_features=ffm_linear_features,
    cross_features=ffm_cross_features,
    embed_dim=10,
    reduction_ratio=1,
    mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| linear_features | list | First-order linear features, usually with `embed_dim=1` | Required |
| cross_features | list | FFM interaction features. Reserve vocabulary space for field offsets. | Required |
| embed_dim | int | FFM embedding width | Required |
| reduction_ratio | int | CEN reduction ratio for FatDeepFFM | Required for FatDeepFFM |
| mlp_params | dict | MLP parameters after the FFM interactions | Required |

## 9. BST

BST appends the candidate item to the user’s history. A Transformer encodes the history and candidate together to score the impression.

### Paper

```
Chen, Qiwei, et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba." arXiv preprint arXiv:1905.06874 (2019).
```

### How it works

- **Transformer encoder.** Multi-head self-attention captures dependencies in the sequence.
- **Positional encoding.** Position information preserves the order of interactions.
- **Combining features.** The model combines the sequence representation with the other features to produce its prediction.

### Usage

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import BST

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id"),
]

model = BST(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    nhead=8,
    dropout=0.2,
    num_layers=1,
    max_seq_len=51,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | User profile and context features, excluding history and target | Required |
| history_features | list | History sequences with `pooling="concat"` | Required |
| target_features | list | Target item features corresponding to the history fields | Required |
| mlp_params | dict | Prediction MLP parameters | Required |
| nhead | int | Attention heads. Must divide the total history embedding width. | 8 |
| dropout | float | Transformer dropout | 0.2 |
| num_layers | int | Transformer encoder layers | 1 |
| max_seq_len | int | Maximum history length plus one position for the target | 51 |

## 10. DIN

DIN uses the candidate to weight the user’s past interactions. The same history can produce different representations for a comedy candidate and a science fiction candidate.

### Paper

```
Zhou, Guorui, et al. "Deep interest network for click-through rate prediction." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

Place the current candidate in `target_features` and the corresponding sequences in `history_features`. If both contain IDs from the same item vocabulary, `shared_with` lets them use the same embedding table.

### Usage

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIN

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
target_features = [SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id")
]

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    attention_mlp_params={"dims": [64, 32], "use_softmax": False},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | User profile and context features | Required |
| history_features | list | History sequences with `pooling="concat"` | Required |
| target_features | list | Target features matching the number, order and widths of the history features | Required |
| mlp_params | dict | Prediction MLP parameters | Required |
| attention_mlp_params | dict | Activation unit parameters | Required |

## 11. DIEN

DIEN reads the history with a GRU and uses the candidate item to determine which changes in interest matter. Training also needs a negative history sequence to supervise the interest representation at each step.

### Paper

```
Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI conference on artificial intelligence. 2019.
```

### How it works

- **Interest extraction.** A GRU encodes the behavior sequence. An auxiliary loss uses positive and negative examples to supervise each hidden state (Eq. 7 in the paper).
- **Interest evolution.** AUGRU incorporates attention scores in its update gate. Attention is normalized over the full sequence with softmax before the recurrent updates (Eqs. 14–16).
- **Auxiliary loss.** $L_{aux} = -\frac{1}{N}\sum[\log\sigma(h_t \cdot e^+_{t+1}) + \log(1-\sigma(h_t \cdot e^-_{t+1}))]$
- **Padding.** The GRU and AUGRU skip padded positions (index 0). An empty history keeps its zero hidden state.

### Usage

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIEN

# Set padding_idx=0 on the target, which owns the embedding shared with both histories.
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items+1, embed_dim=8, padding_idx=0),
]
# Point shared_with on both histories to the target feature (the root key in embed_dict).
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items+1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
]
neg_history_features = [
    SequenceFeature("neg_hist_item_id", vocab_size=n_items+1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
]

model = DIEN(
    features=features,
    history_features=history_features,
    neg_history_features=neg_history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    alpha=0.2,
)
# Use loss_mode=False: forward returns (prediction, aux_loss).
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| features | list | User profile and context features passed to the top MLP |
| history_features | list | Positive history sequences with pooling="concat", padding_idx=0 and shared_with set to the target feature |
| neg_history_features | list | Negative history sequences with the same settings. Point shared_with to the target feature itself. |
| target_features | list | Target item features used by AUGRU attention, with padding_idx=0 |
| mlp_params | dict | Top MLP parameters. The activation is fixed to dice. |
| alpha | float | Auxiliary loss weight, default 0.2 |

## 12. AutoInt

AutoInt applies self-attention across fields within an example. For instance, a movie genre can update its representation using the user’s age and the time of the impression.

### Paper

```
Song, Weiping, et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

Numerical fields are projected into the embedding space with a linear transformation. Multi-head attention then operates across numerical and categorical fields of equal width.

### Usage

```python
from torch_rechub.models.ranking import AutoInt

# Create the model
model = AutoInt(
    sparse_features=sparse_features,
    dense_features=dense_features,
    num_layers=3,
    num_heads=2,
    dropout=0.2,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| sparse_features | list | One or more sparse features with the same `embed_dim` | Required |
| dense_features | list | Continuous features. May be an empty list. | Required |
| num_layers | int | Number of interacting layers | 3 |
| num_heads | int | Attention heads | 2 |
| dropout | float | Attention dropout | 0.0 |
| mlp_params | dict or None | Optional deep branch parameters | None |

## 13. Choosing a retrieval model

Choose a ranking model based on the fields available for one impression. With tabular user, item and context features, start with WideDeep or DeepFM and compare DCN to test explicit feature crosses. If you have chronological user behavior, try DIN, DIEN and BST. All three use a candidate and historical items. DIEN also needs a negative item at each historical position during training.

Keep the training set, validation set and feature encoding fixed when comparing models. Compute AUC and LogLoss on the same validation rows to see the effect of the model change. [Interpreting ranking metrics](/blog/rank)

## 14. Run a Criteo baseline

The [quick start](/guide/quick_start) covers splitting the raw data, encoding features, training DeepFM and evaluating it. To run the training script from the repository root:

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 1 --device cpu
```

## 15. Training questions

**Why is validation AUC falling while training loss is still decreasing?** They measure different things. Compare AUC from earlier epochs on the same validation set. When `CTRTrainer.fit` receives a validation set, it saves the weights from the epoch with the best validation AUC. Use those weights for prediction.

**How do I use a GPU?** For one GPU, pass `device="cuda:0"` to `CTRTrainer`. On a multi-GPU machine, `gpus=[0, 1]` enables PyTorch `DataParallel`. Choose a batch size that fits the memory of each GPU.
