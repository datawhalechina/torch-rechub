---
title: Ranking Models
description: Torch-RecHub ranking models detailed introduction
---

# Ranking Models

Ranking models are core components in recommendation systems, used to predict users' click-through rates or preference scores for items, thereby performing fine-grained ranking on retrieved candidates. Torch-RecHub provides various advanced ranking models covering different feature processing and modeling approaches.

## 1. WideDeep

### Description

WideDeep is a hybrid model combining a linear model (Wide part) and a deep neural network (Deep part), designed to leverage both the memorization capability of linear models and the generalization capability of deep models.

### Paper Reference

```
Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
```

### Core Principles

- **Wide Part**: Linear model using cross features, good at capturing memorization effects
- **Deep Part**: Deep neural network using Embedding and fully connected layers, good at capturing generalization effects
- **Joint Training**: Wide and Deep parts are trained simultaneously, outputs combined through sigmoid function

### Usage

```python
from torch_rechub.models.ranking import WideDeep

dense_features = [DenseFeature(name="age", embed_dim=1), DenseFeature(name="income", embed_dim=1)]
sparse_features = [SparseFeature(name="city", vocab_size=100, embed_dim=16), SparseFeature(name="gender", vocab_size=3, embed_dim=8)]

model = WideDeep(
    wide_features=sparse_features,
    deep_features=sparse_features + dense_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| wide_features | list | Feature list for Wide part | None |
| deep_features | list | Feature list for Deep part | None |
| mlp_params | dict | DNN parameters including dims, dropout, activation | None |

### Use Cases

- Basic ranking tasks
- Scenarios requiring both memorization and generalization
- Limited feature engineering resources

## 2. DeepFM

### Description

DeepFM combines Factorization Machine (FM) and deep neural network, capable of capturing both low-order and high-order feature interactions.

### Paper Reference

```
Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." Proceedings of the 26th international joint conference on artificial intelligence. 2017.
```

### Core Principles

- **FM Part**: Captures second-order feature interactions with linear complexity
- **Deep Part**: Captures high-order feature interactions through neural network
- **Shared Embedding**: FM and Deep parts share feature embeddings, reducing parameters

### Usage

```python
from torch_rechub.models.ranking import DeepFM

model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| fm_features | list | Feature list for FM part | None |
| mlp_params | dict | DNN parameters | None |

### Use Cases

- Scenarios where feature interactions are important
- Need to capture both low-order and high-order feature interactions
- CTR prediction tasks

## 3. DCN

### Description

DCN (Deep & Cross Network) explicitly learns feature crosses through a Cross Network while maintaining linear computational complexity.

### Paper Reference

```
Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017.
```

### Core Principles

- **Cross Network**: Explicitly learns high-order feature crosses, each layer output:
  $$x_{l+1} = x_0 x_l^T w_l + b_l + x_l$$
- **Deep Network**: Deep neural network capturing nonlinear feature interactions
- **Joint Training**: Cross and Deep networks compute in parallel, results concatenated for final output

### Usage

```python
from torch_rechub.models.ranking import DCN

model = DCN(
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| cross_features | list | Feature list for Cross part | None |
| mlp_params | dict | DNN parameters | None |
| cross_num_layers | int | Number of Cross Network layers | 3 |

### Use Cases

- Scenarios requiring explicit feature crosses
- Limited computational resources
- CTR prediction tasks

## 4. DCNv2

### Description

DCNv2 is an enhanced version of DCN, introducing feature selection units and dynamic scaling mechanisms for improved expressiveness and efficiency.

### Paper Reference

```
Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021.
```

### Core Principles

- **Feature Selection Unit**: Assigns dynamic weights to each feature, automatically selecting important features
- **Dynamic Scaling**: Introduces scalar parameters to adaptively adjust feature cross contributions
- **Flexible Cross Network**: Supports different cross forms

### Usage

```python
from torch_rechub.models.ranking import DCNv2

model = DCNv2(
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| cross_features | list | Feature list for Cross part | None |
| mlp_params | dict | DNN parameters | None |
| cross_num_layers | int | Number of Cross Network layers | 3 |

### Use Cases

- Scenarios requiring more efficient feature crosses
- Large-scale recommendation systems
- CTR prediction tasks

## 5. EDCN

### Description

EDCN (Enhanced Deep & Cross Network) is an enhanced cross network model that combines explicit feature crosses with deep feature extraction for improved expressiveness.

### Paper Reference

```
Ma, Xiao, et al. "Enhanced Deep & Cross Network for Feature Cross Learning in Click-Through Rate Prediction." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

### Core Principles

- **Cross Network**: Explicitly learns high-order feature crosses
- **Deep Network**: Deep neural network capturing nonlinear feature interactions
- **Feature Importance Learning**: Introduces feature importance weights for better interpretability

### Usage

```python
from torch_rechub.models.ranking import EDCN

model = EDCN(
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| cross_features | list | Feature list for Cross part | None |
| mlp_params | dict | DNN parameters | None |
| cross_num_layers | int | Number of Cross Network layers | 3 |

### Use Cases

- Complex feature interaction scenarios
- Models requiring high expressiveness
- CTR prediction tasks

## 6. AFM

### Description

AFM (Attention Factorization Machine) is an attention-based factorization machine that adaptively learns the importance of different feature interactions.

### Paper Reference

```
Xiao, Jun, et al. "Attentional factorization machines: Learning the weight of feature interactions via attention networks." arXiv preprint arXiv:1708.04617 (2017).
```

### Core Principles

- **FM Foundation**: Based on factorization machine, captures second-order feature interactions
- **Attention Mechanism**: Introduces attention network to assign dynamic weights to each feature interaction
- **Attention Output**: Weighted sum of attention weights and feature interaction vectors

### Usage

```python
from torch_rechub.models.ranking import AFM

model = AFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    attention_params={"attention_dim": 64, "dropout": 0.2}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| fm_features | list | Feature list for FM part | None |
| attention_params | dict | Attention network parameters | None |

### Use Cases

- Scenarios with varying feature interaction importance
- Need for interpretability
- CTR prediction tasks

## 7. FiBiNET

### Description

FiBiNET (Feature Importance and Bilinear feature Interaction NETwork) combines feature importance learning with bilinear feature interactions for more effective feature interaction capture.

### Paper Reference

```
Juan, Yuchin, et al. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

### Core Principles

- **Feature Importance Network**: Learns feature importance through Squeeze-and-Excitation mechanism
- **Bilinear Interaction**: Uses bilinear functions to capture feature interactions
- **Feature Enhancement**: Enhances input features for improved expressiveness

### Usage

```python
from torch_rechub.models.ranking import FiBiNet

model = FiBiNet(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| fm_features | list | Feature list for FM part | None |
| mlp_params | dict | DNN parameters | None |

### Use Cases

- Scenarios with varying feature importance
- Need for complex feature interactions
- CTR prediction tasks

## 8. DeepFFM

### Description

DeepFFM (Deep Field-aware Factorization Machine) combines field-aware factorization machine with deep neural network, capturing field-aware high-order feature interactions.

### Paper Reference

```
Xiao, Jun, et al. "Deep learning over multi-field categorical data." European conference on information retrieval. Springer, Cham, 2016.
```

### Core Principles

- **FFM Foundation**: Field-aware factorization machine, learns specific interaction vectors for each field pair
- **Deep Network**: Deep neural network capturing high-order feature interactions
- **Joint Training**: FFM and Deep parts trained jointly

### Usage

```python
from torch_rechub.models.ranking import DeepFFM, FatDeepFFM

model = DeepFFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# FatDeepFFM (enhanced version)
fat_model = FatDeepFFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| fm_features | list | Feature list for FFM part | None |
| mlp_params | dict | DNN parameters | None |

### Use Cases

- Scenarios where field-aware feature interactions are important
- Complex feature interaction scenarios
- CTR prediction tasks

## 9. BST

### Description

BST (Behavior Sequence Transformer) uses Transformer to model user behavior sequences, capturing long-range dependencies in sequences.

### Paper Reference

```
Sun, Fei, et al. "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.
```

### Core Principles

- **Transformer Encoder**: Uses multi-head self-attention to capture sequence dependencies
- **Positional Encoding**: Adds positional information to preserve sequence order
- **Feature Fusion**: Fuses sequence features with other features for final prediction

### Usage

```python
from torch_rechub.models.ranking import BST

sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling="mean")]

model = BST(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    transformer_params={"num_heads": 4, "num_layers": 2, "hidden_size": 128, "dropout": 0.2}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| sequence_features | list | Sequence feature list | None |
| transformer_params | dict | Transformer parameters | None |

### Use Cases

- Scenarios where user behavior sequences are important
- Long sequence modeling
- Sequential recommendation tasks

## 10. DIN

### Description

DIN (Deep Interest Network) is an attention-based deep interest network that dynamically captures user interests based on target items.

### Paper Reference

```
Zhou, Guorui, et al. "Deep interest network for click-through rate prediction." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### Core Principles

- **Interest Extraction**: Extracts interest representations from user behavior sequences
- **Attention Mechanism**: Computes attention weights for each historical behavior based on target item
- **Dynamic Interest Aggregation**: Dynamically aggregates user interests based on attention weights

### Usage

```python
from torch_rechub.models.ranking import DIN

sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling=None)]

model = DIN(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    target_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| sequence_features | list | Sequence feature list | None |
| target_features | list | Target feature list | None |
| mlp_params | dict | DNN parameters | None |

### Use Cases

- Scenarios with dynamic user interests
- Target item-related interest modeling
- CTR prediction tasks

## 11. DIEN

### Description

DIEN (Deep Interest Evolution Network) models user interest evolution, capturing dynamic changes in user interests.

### Paper Reference

```
Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI conference on artificial intelligence. 2019.
```

### Core Principles

- **GRU**: Uses GRU to capture temporal changes in user interests
- **Interest Extraction Layer**: Extracts interest sequences from raw behavior sequences
- **Interest Evolution Layer**: Models dynamic interest evolution using GRU and attention
- **Interest Activation Layer**: Activates relevant interests based on target item

### Usage

```python
from torch_rechub.models.ranking import DIEN

sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling=None)]

model = DIEN(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    target_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    dien_params={"gru_layers": 2, "attention_dim": 64, "dropout": 0.2}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| sequence_features | list | Sequence feature list | None |
| target_features | list | Target feature list | None |
| mlp_params | dict | DNN parameters | None |
| dien_params | dict | DIEN network parameters | None |

### Use Cases

- Scenarios with evolving user interests
- Long sequence interest modeling
- CTR prediction tasks

## 12. AutoInt

### Description

AutoInt (Automatic Feature Interaction Learning via Self-Attentive Neural Networks) uses self-attention to automatically learn feature interactions, flexibly capturing various orders of feature interactions.

### Paper Reference

```
Song, Weiping, et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

### Core Principles

- **Embedding Layer**: Maps discrete features to low-dimensional vector space
- **Multi-head Self-attention**: Automatically learns interaction relationships between features
- **Residual Connection**: Enhances training stability
- **Layer Normalization**: Accelerates model convergence

### Usage

```python
from torch_rechub.models.ranking import AutoInt

model = AutoInt(
    deep_features=sparse_features + dense_features,
    attention_params={"num_heads": 4, "num_layers": 2, "hidden_size": 128, "dropout": 0.2}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| deep_features | list | Feature list for Deep part | None |
| attention_params | dict | Attention network parameters | None |

### Use Cases

- Automatic feature interaction learning
- Complex feature interaction scenarios
- CTR prediction tasks

## 13. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Interpretability |
| --- | --- | --- | --- | --- |
| WideDeep | Low | Medium | High | High |
| DeepFM | Medium | High | Medium | Medium |
| DCN/DCNv2 | Medium | High | High | Medium |
| EDCN | Medium | High | Medium | Medium |
| AFM | Medium | Medium | Medium | High |
| FiBiNET | Medium | High | Medium | Medium |
| DeepFFM | High | High | Low | Medium |
| BST | High | High | Low | Medium |
| DIN | Medium | High | Medium | Medium |
| DIEN | High | High | Low | Medium |
| AutoInt | High | High | Low | Medium |

## 14. Usage Recommendations

1. **Choose based on data scale**: For small-scale data, use simple models (WideDeep, DeepFM); for large-scale data, try more complex models
2. **Choose based on feature types**: For important sequence features, use BST, DIN, DIEN; for important feature interactions, use DCN, DeepFM
3. **Choose based on computational resources**: For limited resources, use efficient models (DCN, WideDeep)
4. **Try multiple models and ensemble**: Different models may capture different feature interaction patterns; ensembling can improve results

## 15. Complete Training Example

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import DenseFeature, SparseFeature

# 1. Define features
dense_features = [
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

sparse_features = [
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8),
    SparseFeature(name="occupation", vocab_size=20, embed_dim=12)
]

# 2. Prepare data
x = {
    "age": age_data,
    "income": income_data,
    "city": city_data,
    "gender": gender_data,
    "occupation": occupation_data
}
y = label_data

# 3. Create data generator
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 4. Create model
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 5. Create trainer
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/deepfm"
)

# 6. Train model
trainer.fit(train_dl, val_dl)

# 7. Evaluate model
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc}")

# 8. Export ONNX model
trainer.export_onnx("deepfm.onnx")
```

## 16. FAQ

### Q: How to choose the right model?
A: Choose based on data scale, feature types, computational resources, and business requirements. Start with simple models and gradually try more complex ones.

### Q: What to do about overfitting?
A: Try the following:
- Add regularization (L1/L2)
- Increase dropout rate
- Use early stopping
- Add more training data
- Simplify model structure

### Q: How to handle large-scale features?
A: Try the following:
- Feature selection: Keep only important features
- Feature hashing: Map high-dimensional features to low-dimensional space
- Hierarchical embedding: Use different embedding dimensions for different features

### Q: How to speed up training?
A: Try the following:
- Use GPU training
- Increase batch size
- Use mixed precision training
- Choose efficient models
- Data parallel training
