# torch-rechub API 文档

> 基于 AST 扫描 `torch_rechub/` 自动生成。已排除 `__init__.py` 文件。

## `basic/`

### `activation`

模块: `torch_rechub.basic.activation`

#### `activation_layer`

```python
activation_layer(act_name)
```

Construct activation layers

**参数**

- `act_name` (`str or nn.Module, name of activation function`)

**返回**

- `act_layer` (`activation layer`)

#### `Dice`

The Dice activation function mentioned in the `DIN paper
https://arxiv.org/abs/1706.06978`

##### `Dice.forward`

```python
forward(self, x: torch.Tensor)
```

未提供文档说明。

### `callback`

模块: `torch_rechub.basic.callback`

#### `EarlyStopper`

Early stops the training if validation loss doesn't improve after a given patience.

**参数**

- `patience` (`int`): How long to wait after last time validation auc improved.

##### `EarlyStopper.stop_training`

```python
stop_training(self, val_auc, weights)
```

whether to stop training.

**参数**

- `val_auc` (`float`): auc score in val data.
- `weights` (`tensor`): the weights of model

### `features`

模块: `torch_rechub.basic.features`

#### `SequenceFeature`

The Feature Class for Sequence feature or multi-hot feature.
In recommendation, there are many user behaviour features which we want to take the sequence model
and tag featurs (multi hot) which we want to pooling. Note that if you use this feature, you must padding
the feature value before training.

**参数**

- `name` (`str`): feature's name.
- `vocab_size` (`int`): vocabulary size of embedding table.
- `embed_dim` (`int`): embedding vector's length
- `pooling` (`str`): pooling method, support `["mean", "sum", "concat"]` (default=`"mean"`)
- `shared_with` (`str`): the another feature name which this feature will shared with embedding.
- `padding_idx` (`int, optional`): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
- `initializer` (`Initializer`): Initializer the embedding layer weight.

##### `SequenceFeature.get_embedding_layer`

```python
get_embedding_layer(self)
```

未提供文档说明。

#### `SparseFeature`

The Feature Class for Sparse feature.

**参数**

- `name` (`str`): feature's name.
- `vocab_size` (`int`): vocabulary size of embedding table.
- `embed_dim` (`int`): embedding vector's length
- `shared_with` (`str`): the another feature name which this feature will shared with embedding.
- `padding_idx` (`int, optional`): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
- `initializer` (`Initializer`): Initializer the embedding layer weight.

##### `SparseFeature.get_embedding_layer`

```python
get_embedding_layer(self)
```

未提供文档说明。

#### `DenseFeature`

The Feature Class for Dense feature.

**参数**

- `name` (`str`): feature's name.
- `embed_dim` (`int`): embedding vector's length, the value fixed `1`. If you put a vector (torch.tensor) , replace the embed_dim with your vector dimension.

### `initializers`

模块: `torch_rechub.basic.initializers`

#### `RandomNormal`

Returns an embedding initialized with a normal distribution.

**参数**

- `mean` (`float`): the mean of the normal distribution
- `std` (`float`): the standard deviation of the normal distribution

#### `RandomUniform`

Returns an embedding initialized with a uniform distribution.

**参数**

- `minval` (`float`): Lower bound of the range of random values of the uniform distribution.
- `maxval` (`float`): Upper bound of the range of random values of the uniform distribution.

#### `XavierNormal`

Returns an embedding initialized with  the method described in
`Understanding the difficulty of training deep feedforward neural networks`
- Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

**参数**

- `gain` (`float`): stddev = gain*sqrt(2 / (fan_in + fan_out))

#### `XavierUniform`

Returns an embedding initialized with the method described in
`Understanding the difficulty of training deep feedforward neural networks`
- Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

**参数**

- `gain` (`float`): stddev = gain*sqrt(6 / (fan_in + fan_out))

#### `Pretrained`

Creates Embedding instance from given 2-dimensional FloatTensor.

**参数**

- `embedding_weight` (`Tensor or ndarray or List[List[int]]`): FloatTensor containing weights for the Embedding.
  First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
- `freeze` (`boolean, optional`): If ``True``, the tensor does not get updated in the learning process.

### `layers`

模块: `torch_rechub.basic.layers`

#### `PredictionLayer`

Prediction layer.

**参数**

- `task_type` (`{'classification', 'regression'}`): Classification applies sigmoid to logits; regression returns logits.

##### `PredictionLayer.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `EmbeddingLayer`

General embedding layer.

Stores per-feature embedding tables in ``embed_dict``.

**参数**

- `features` (`list`): Feature objects to create embedding tables for.

**张量形状**

- **Input**
  - `x` (`dict`): ``{feature_name: feature_value}``; sequence values shape ``(B, L)``,
    sparse/dense values shape ``(B,)``.
  - `features` (`list`): Feature list for lookup.
  - `squeeze_dim` (`bool, default False`): Whether to flatten embeddings.
- **Output**
  - Dense only: ``(B, num_dense)``.
  - Sparse: ``(B, num_features, embed_dim)`` or flattened.
  - Sequence: same as sparse or ``(B, num_seq, L, embed_dim)`` when ``pooling="concat"``.
  - Mixed: flattened sparse plus dense when ``squeeze_dim=True``.

##### `EmbeddingLayer.forward`

```python
forward(self, x, features, squeeze_dim = False)
```

未提供文档说明。

#### `InputMask`

Return input masks from features.

**张量形状**

- **Input**
  - `x` (`dict`): ``{feature_name: feature_value}``; sequence ``(B, L)``, sparse/dense ``(B,)``.
  - `features` (`list or SparseFeature or SequenceFeature`): All elements must be sparse or sequence features.
- **Output**
  - Sparse: ``(B, num_features)``
  - Sequence: ``(B, num_seq, seq_length)``

##### `InputMask.forward`

```python
forward(self, x, features)
```

未提供文档说明。

#### `LR`

Logistic regression module.

**参数**

- `input_dim` (`int`): Input dimension.
- `sigmoid` (`bool, default False`): Apply sigmoid to output when True.

**张量形状**

Input: ``(B, input_dim)``
Output: ``(B, 1)``

##### `LR.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `ConcatPooling`

Keep original sequence embedding shape.

**张量形状**

Input: ``(B, L, D)``  
Output: ``(B, L, D)``

##### `ConcatPooling.forward`

```python
forward(self, x, mask = None)
```

未提供文档说明。

#### `AveragePooling`

Mean pooling over sequence embeddings.

**张量形状**

- **Input**
  - `x` (```(B, L, D)```)
  - `mask` (```(B, 1, L)```)
- **Output**
  ``(B, D)``

##### `AveragePooling.forward`

```python
forward(self, x, mask = None)
```

未提供文档说明。

#### `SumPooling`

Sum pooling over sequence embeddings.

**张量形状**

- **Input**
  - `x` (```(B, L, D)```)
  - `mask` (```(B, 1, L)```)
- **Output**
  ``(B, D)``

##### `SumPooling.forward`

```python
forward(self, x, mask = None)
```

未提供文档说明。

#### `MLP`

Multi-layer perceptron with BN/activation/dropout per linear layer.

**参数**

- `input_dim` (`int`): Input dimension of the first linear layer.
- `output_layer` (`bool, default True`): If True, append a final Linear(*,1).
- `dims` (`list, default []`): Hidden layer sizes.
- `dropout` (`float, default 0`): Dropout probability.
- `activation` (`str, default 'relu'`): Activation function (sigmoid, relu, prelu, dice, softmax).

**张量形状**

Input: ``(B, input_dim)``  
Output: ``(B, 1)`` or ``(B, dims[-1])``

##### `MLP.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `FM`

Factorization Machine for 2nd-order interactions.

**参数**

- `reduce_sum` (`bool, default True`): Sum over embed dim (inner product) when True; otherwise keep dim.

**张量形状**

Input: ``(B, num_features, embed_dim)``  
Output: ``(B, 1)`` or ``(B, embed_dim)``

##### `FM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `CIN`

Compressed Interaction Network.

**参数**

- `input_dim` (`int`): Input dimension.
- `cin_size` (`list[int]`): Output channels per Conv1d layer.
- `split_half` (`bool, default True`): Split channels except last layer.

**张量形状**

Input: ``(B, num_features, embed_dim)``  
Output: ``(B, 1)``

##### `CIN.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `CrossLayer`

Cross layer.

**参数**

- `input_dim` (`int`): Input dimension.

##### `CrossLayer.forward`

```python
forward(self, x_0, x_i)
```

未提供文档说明。

#### `CrossNetwork`

CrossNetwork from DCN.

**参数**

- `input_dim` (`int`): Input dimension.
- `num_layers` (`int`): Number of cross layers.

**张量形状**

Input: ``(B, *)``  
Output: ``(B, *)``

##### `CrossNetwork.forward`

```python
forward(self, x)
```

:param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``

#### `CrossNetV2`

DCNv2-style cross network.

**参数**

- `input_dim` (`int`): Input dimension.
- `num_layers` (`int`): Number of cross layers.

##### `CrossNetV2.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `CrossNetMix`

CrossNetMix with MOE and nonlinear low-rank transforms.

**说明**

Input: float tensor ``(B, num_fields, embed_dim)``.

##### `CrossNetMix.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `SENETLayer`

SENet-style feature gating.

**参数**

- `num_fields` (`int`): Number of feature fields.
- `reduction_ratio` (`int, default=3`): Reduction ratio for the bottleneck MLP.

##### `SENETLayer.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `BiLinearInteractionLayer`

Bilinear feature interaction (FFM-style).

**参数**

- `input_dim` (`int`): Input dimension.
- `num_fields` (`int`): Number of feature fields.
- `bilinear_type` (`{'field_all', 'field_each', 'field_interaction'}, default 'field_interaction'`): Bilinear interaction variant.

##### `BiLinearInteractionLayer.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `MultiInterestSA`

Self-attention multi-interest module (Comirec).

**参数**

- `embedding_dim` (`int`): Item embedding dimension.
- `interest_num` (`int`): Number of interests.
- `hidden_dim` (`int, optional`): Hidden dimension; defaults to ``4 * embedding_dim`` if None.

**张量形状**

- **Input**
  - `seq_emb` (```(B, L, D)```)
  - `mask` (```(B, L, 1)```)
- **Output**
  ``(B, interest_num, D)``

##### `MultiInterestSA.forward`

```python
forward(self, seq_emb, mask = None)
```

未提供文档说明。

#### `CapsuleNetwork`

Capsule network for multi-interest (MIND/Comirec).

**参数**

- `embedding_dim` (`int`): Item embedding dimension.
- `seq_len` (`int`): Sequence length.
- `bilinear_type` (`{0, 1, 2}, default 2`): 0 for MIND, 2 for ComirecDR.
- `interest_num` (`int, default 4`): Number of interests.
- `routing_times` (`int, default 3`): Routing iterations.
- `relu_layer` (`bool, default False`): Whether to apply ReLU after routing.

**张量形状**

- **Input**
  - `seq_emb` (```(B, L, D)```)
  - `mask` (```(B, L, 1)```)
- **Output**
  ``(B, interest_num, D)``

##### `CapsuleNetwork.forward`

```python
forward(self, item_eb, mask)
```

未提供文档说明。

#### `FFM`

The Field-aware Factorization Machine module, mentioned in the `FFM paper
<https://dl.acm.org/doi/abs/10.1145/2959100.2959134>`. It explicitly models
multi-channel second-order feature interactions, with each feature filed
corresponding to one channel.

**参数**

- `num_fields` (`int`): number of feature fields.
- `reduce_sum` (`bool`): whether to sum in embed_dim (default = `True`).

**张量形状**

    - Input: `(batch_size, num_fields, num_fields, embed_dim)`
    - Output: `(batch_size, num_fields*(num_fields-1)/2, 1)` or `(batch_size, num_fields*(num_fields-1)/2, embed_dim)`

##### `FFM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `CEN`

The Compose-Excitation Network module, mentioned in the `FAT-DeepFFM paper
<https://arxiv.org/abs/1905.06336>`, a modified version of
`Squeeze-and-Excitation Network” (SENet) (Hu et al., 2017)`. It is used to
highlight the importance of second-order feature crosses.

**参数**

- `embed_dim` (`int`): the dimensionality of categorical value embedding.
- `num_field_crosses` (`int`): the number of second order crosses between feature fields.
- `reduction_ratio` (`int`): the between the dimensions of input layer and hidden layer of the MLP module.

**张量形状**

    - Input: `(batch_size, num_fields, num_fields, embed_dim)`
    - Output: `(batch_size, num_fields*(num_fields-1)/2 * embed_dim)`

##### `CEN.forward`

```python
forward(self, em)
```

未提供文档说明。

#### `HSTULayer`

Single HSTU layer.

This layer implements the core HSTU "sequential transduction unit": a
multi-head self-attention block with gating and a position-wise FFN, plus
residual connections and LayerNorm.

**参数**

- `d_model` (`int`): Hidden dimension of the model. Default: 512.
- `n_heads` (`int`): Number of attention heads. Default: 8.
- `dqk` (`int`): Dimension of query/key per head. Default: 64.
- `dv` (`int`): Dimension of value per head. Default: 64.
- `dropout` (`float`): Dropout rate applied in the layer. Default: 0.1.
- `use_rel_pos_bias` (`bool`): Whether to use relative position bias.

**张量形状**

    - Input: ``(batch_size, seq_len, d_model)``
    - Output: ``(batch_size, seq_len, d_model)``

**示例**

```python
>>> layer = HSTULayer(d_model=512, n_heads=8)
>>> x = torch.randn(32, 256, 512)
>>> output = layer(x)
>>> output.shape
```
    torch.Size([32, 256, 512])

##### `HSTULayer.forward`

```python
forward(self, x, rel_pos_bias = None)
```

Forward pass of a single HSTU layer.

**参数**

- `x` (`Tensor`): Input tensor of shape ``(batch_size, seq_len, d_model)``.
- `rel_pos_bias` (`Tensor, optional`): Relative position bias of shape
  ``(1, n_heads, seq_len, seq_len)``.

**返回**

- `Tensor` (`Output tensor of shape ``(batch_size, seq_len, d_model)``.`)

#### `HSTUBlock`

Stacked HSTU block.

This block stacks multiple :class:`HSTULayer` layers to form a deep HSTU
encoder for sequential recommendation.

**参数**

- `d_model` (`int`): Hidden dimension of the model. Default: 512.
- `n_heads` (`int`): Number of attention heads. Default: 8.
- `n_layers` (`int`): Number of stacked HSTU layers. Default: 4.
- `dqk` (`int`): Dimension of query/key per head. Default: 64.
- `dv` (`int`): Dimension of value per head. Default: 64.
- `dropout` (`float`): Dropout rate applied in each layer. Default: 0.1.
- `use_rel_pos_bias` (`bool`): Whether to use relative position bias.

**张量形状**

    - Input: ``(batch_size, seq_len, d_model)``
    - Output: ``(batch_size, seq_len, d_model)``

**示例**

```python
>>> block = HSTUBlock(d_model=512, n_heads=8, n_layers=4)
>>> x = torch.randn(32, 256, 512)
>>> output = block(x)
>>> output.shape
```
    torch.Size([32, 256, 512])

##### `HSTUBlock.forward`

```python
forward(self, x, rel_pos_bias = None)
```

Forward pass through all stacked HSTULayer modules.

**参数**

- `x` (`Tensor`): Input tensor of shape ``(batch_size, seq_len, d_model)``.
- `rel_pos_bias` (`Tensor, optional`): Relative position bias shared across
  all layers.

**返回**

- `Tensor` (`Output tensor of shape ``(batch_size, seq_len, d_model)``.`)

#### `InteractingLayer`

Multi-head Self-Attention based Interacting Layer, used in AutoInt model.

**参数**

- `embed_dim` (`int`): the embedding dimension.
- `num_heads` (`int`): the number of attention heads (default=2).
- `dropout` (`float`): the dropout rate (default=0.0).
- `residual` (`bool`): whether to use residual connection (default=True).

**张量形状**

    - Input: `(batch_size, num_fields, embed_dim)`
    - Output: `(batch_size, num_fields, embed_dim)`

##### `InteractingLayer.forward`

```python
forward(self, x)
```

**参数**

- `x` (`input tensor with shape (batch_size, num_fields, embed_dim)`)

### `loss_func`

模块: `torch_rechub.basic.loss_func`

#### `RegularizationLoss`

Unified L1/L2 regularization for embedding and dense parameters.

**参数**

- `embedding_l1` (`float, default=0.0`): L1 coefficient for embedding parameters.
- `embedding_l2` (`float, default=0.0`): L2 coefficient for embedding parameters.
- `dense_l1` (`float, default=0.0`): L1 coefficient for dense (non-embedding) parameters.
- `dense_l2` (`float, default=0.0`): L2 coefficient for dense (non-embedding) parameters.

**示例**

```python
>>> reg_loss_fn = RegularizationLoss(embedding_l2=1e-5, dense_l2=1e-5)
>>> reg_loss = reg_loss_fn(model)
>>> total_loss = task_loss + reg_loss
```

##### `RegularizationLoss.forward`

```python
forward(self, model)
```

未提供文档说明。

#### `HingeLoss`

Hinge loss for pairwise learning.

**说明**

Reference: https://github.com/ustcml/RecStudio/blob/main/recstudio/model/loss_func.py

##### `HingeLoss.forward`

```python
forward(self, pos_score, neg_score, in_batch_neg = False)
```

未提供文档说明。

#### `BPRLoss`

未提供文档说明。

##### `BPRLoss.forward`

```python
forward(self, pos_score, neg_score, in_batch_neg = False)
```

未提供文档说明。

#### `NCELoss`

Noise Contrastive Estimation (NCE) loss for recommender systems.

**参数**

- `temperature` (`float, default=1.0`): Temperature for scaling logits.
- `ignore_index` (`int, default=0`): Target index to ignore.
- `reduction` (`{'mean', 'sum', 'none'}, default='mean'`): Reduction applied to the output.

**说明**

- Gutmann & Hyvärinen (2010), Noise-contrastive estimation.
- HLLM: Hierarchical Large Language Model for Recommendation.

**示例**

```python
>>> nce_loss = NCELoss(temperature=0.1)
>>> logits = torch.randn(32, 1000)
>>> targets = torch.randint(0, 1000, (32,))
>>> loss = nce_loss(logits, targets)
```

##### `NCELoss.forward`

```python
forward(self, logits, targets)
```

Compute NCE loss.

**参数**

- `logits` (`torch.Tensor`): Model output logits of shape (batch_size, vocab_size)
- `targets` (`torch.Tensor`): Target indices of shape (batch_size,)

**返回**

- `torch.Tensor` (`NCE loss value`)

#### `InBatchNCELoss`

In-batch NCE loss with explicit negatives.

**参数**

- `temperature` (`float, default=0.1`): Temperature for scaling logits.
- `ignore_index` (`int, default=0`): Target index to ignore.
- `reduction` (`{'mean', 'sum', 'none'}, default='mean'`): Reduction applied to the output.

**示例**

```python
>>> loss_fn = InBatchNCELoss(temperature=0.1)
>>> embeddings = torch.randn(32, 256)
>>> item_embeddings = torch.randn(1000, 256)
>>> targets = torch.randint(0, 1000, (32,))
>>> loss = loss_fn(embeddings, item_embeddings, targets)
```

##### `InBatchNCELoss.forward`

```python
forward(self, embeddings, item_embeddings, targets)
```

Compute in-batch NCE loss.

**参数**

- `embeddings` (`torch.Tensor`): User/query embeddings of shape (batch_size, embedding_dim)
- `item_embeddings` (`torch.Tensor`): Item embeddings of shape (vocab_size, embedding_dim)
- `targets` (`torch.Tensor`): Target item indices of shape (batch_size,)

**返回**

- `torch.Tensor` (`In-batch NCE loss value`)

### `metaoptimizer`

模块: `torch_rechub.basic.metaoptimizer`

#### `MetaBalance`

MetaBalance Optimizer
   This method is used to scale the gradient and balance the gradient of each task

**参数**

- `parameters` (`list`): the parameters of model
- `relax_factor` (`float, optional`): the relax factor of gradient scaling (default: 0.7)
- `beta` (`float, optional`): the coefficient of moving average (default: 0.9)

##### `MetaBalance.step`

```python
step(self, losses)
```

_summary_

**参数**

- `losses` (`_type_`): _description_

**异常**

- `RuntimeError` (`_description_`)

### `metric`

模块: `torch_rechub.basic.metric`

#### `auc_score`

```python
auc_score(y_true, y_pred)
```

未提供文档说明。

#### `get_user_pred`

```python
get_user_pred(y_true, y_pred, users)
```

divide the result into different group by user id

**参数**

- `y_true` (`array`): all true labels of the data
- `y_pred` (`array`): the predicted score
- `users` (`array`): user id

**返回**

- `user_pred` (`dict`): {userid: values}, key is user id and value is the labels and scores of each user

#### `gauc_score`

```python
gauc_score(y_true, y_pred, users, weights = None)
```

compute GAUC

**参数**

- `y_true` (`array`): dim(N, ), all true labels of the data
- `y_pred` (`array`): dim(N, ), the predicted score
- `users` (`array`): dim(N, ), user id
- `weight` (`dict`): {userid: weight_value}, it contains weights for each group.
  if it is None, the weight is equal to the number
  of times the user is recommended

**返回**

- `score` (`float, GAUC`)

#### `ndcg_score`

```python
ndcg_score(y_true, y_pred, topKs = None)
```

未提供文档说明。

#### `hit_score`

```python
hit_score(y_true, y_pred, topKs = None)
```

未提供文档说明。

#### `mrr_score`

```python
mrr_score(y_true, y_pred, topKs = None)
```

未提供文档说明。

#### `recall_score`

```python
recall_score(y_true, y_pred, topKs = None)
```

未提供文档说明。

#### `precision_score`

```python
precision_score(y_true, y_pred, topKs = None)
```

未提供文档说明。

#### `topk_metrics`

```python
topk_metrics(y_true, y_pred, topKs = None)
```

choice topk metrics and compute it
the metrics contains 'ndcg', 'mrr', 'recall', 'precision' and 'hit'

**参数**

- `y_true` (`dict`): {userid, item_ids}, the key is user id and the value is the list that contains the items the user interacted
- `y_pred` (`dict`): {userid, item_ids}, the key is user id and the value is the list that contains the items recommended
- `topKs` (`list or tuple`): if you want to get top5 and top10, topKs=(5, 10)

**返回**

- `results` (`dict`): {metric_name: metric_values}, it contains five metrics, 'ndcg', 'recall', 'mrr', 'hit', 'precision'

#### `log_loss`

```python
log_loss(y_true, y_pred)
```

未提供文档说明。

#### `diversity_score`

```python
diversity_score(y_pred, item_embeddings, topKs = None)
```

Intra-List Diversity (ILD): average pairwise cosine distance within each user's recommendation list.

A higher score means the recommended items are more different from each other,
indicating the model is not just recommending similar items repeatedly.

**参数**

- `y_pred` (`dict`): {userid: [item_ids]}, recommended items per user
- `item_embeddings` (`dict or np.ndarray`): item vectors. If dict: {item_id: np.array};
  if 2D array: indexed by item_id (row = item_id)
- `topKs` (`list or tuple`): e.g. [5, 10]

**返回**

- `results` (`dict`): {'Diversity': ['Diversity@5: 0.xxxx', ...]}

#### `coverage_score`

```python
coverage_score(y_pred, all_items, topKs = None)
```

Catalog Coverage: fraction of all items that appear in at least one user's recommendation list.

A higher score means the model recommends a wider variety of items across all users,
rather than always recommending the same popular items.

**参数**

- `y_pred` (`dict`): {userid: [item_ids]}, recommended items per user
- `all_items` (`set or list`): all unique item ids in the catalog
- `topKs` (`list or tuple`): e.g. [5, 10]

**返回**

- `results` (`dict`): {'Coverage': ['Coverage@5: 0.xxxx', ...]}

#### `novelty_score`

```python
novelty_score(y_pred, item_popularity, topKs = None)
```

Mean Self-Information: measures how "surprising" or niche the recommendations are.

For each recommended item, self-information = -log2(popularity).
Popular items have low self-information; long-tail items have high self-information.
A higher novelty score means the model recommends more niche items.

**参数**

- `y_pred` (`dict`): {userid: [item_ids]}, recommended items per user
- `item_popularity` (`dict`): {item_id: float}, interaction probability of each item
  (e.g. item_count / total_interactions). Values should be in (0, 1].
- `topKs` (`list or tuple`): e.g. [5, 10]

**返回**

- `results` (`dict`): {'Novelty': ['Novelty@5: x.xxxx', ...]}

### `tracking`

模块: `torch_rechub.basic.tracking`

#### `BaseLogger`

Base interface for experiment tracking backends.

**方法**

- `log_metrics(metrics, step=None)`: Record scalar metrics at a given step.
- `log_hyperparams(params)`: Store hyperparameters and run configuration.
- `finish()`: Flush pending logs and release resources.

##### `BaseLogger.log_metrics`

```python
log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None
```

Log metrics to the tracking backend.

**参数**

- `metrics` (`dict of str to Any`): Metric name-value pairs to record.
- `step` (`int, optional`): Explicit global step or epoch index. When ``None``, the backend
  uses its own default step handling.

##### `BaseLogger.log_hyperparams`

```python
log_hyperparams(self, params: Dict[str, Any]) -> None
```

Log experiment hyperparameters.

**参数**

- `params` (`dict of str to Any`): Hyperparameters or configuration values to persist with the run.

##### `BaseLogger.finish`

```python
finish(self) -> None
```

Finalize logging and free any backend resources.

#### `WandbLogger`

Weights & Biases logger implementation.

**参数**

- `project` (`str`): Name of the wandb project to log to.
- `name` (`str, optional`): Display name for the run.
- `config` (`dict, optional`): Initial hyperparameter configuration to record.
- `tags` (`list of str, optional`): Optional tags for grouping runs.
- `notes` (`str, optional`): Long-form notes shown in the run overview.
- `dir` (`str, optional`): Local directory for wandb artifacts and cache.
- `**kwargs` (`dict`): Additional keyword arguments forwarded to ``wandb.init``.

**异常**

- `ImportError`: If ``wandb`` is not installed in the current environment.

##### `WandbLogger.log_metrics`

```python
log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None
```

未提供文档说明。

##### `WandbLogger.log_hyperparams`

```python
log_hyperparams(self, params: Dict[str, Any]) -> None
```

未提供文档说明。

##### `WandbLogger.finish`

```python
finish(self) -> None
```

未提供文档说明。

#### `SwanLabLogger`

SwanLab logger implementation.

**参数**

- `project` (`str, optional`): Project identifier for grouping experiments.
- `experiment_name` (`str, optional`): Display name for the experiment or run.
- `description` (`str, optional`): Text description shown alongside the run.
- `config` (`dict, optional`): Hyperparameters or configuration to log at startup.
- `logdir` (`str, optional`): Directory where logs and artifacts are stored.
- `**kwargs` (`dict`): Additional keyword arguments forwarded to ``swanlab.init``.

**异常**

- `ImportError`: If ``swanlab`` is not installed in the current environment.

##### `SwanLabLogger.log_metrics`

```python
log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None
```

未提供文档说明。

##### `SwanLabLogger.log_hyperparams`

```python
log_hyperparams(self, params: Dict[str, Any]) -> None
```

未提供文档说明。

##### `SwanLabLogger.finish`

```python
finish(self) -> None
```

未提供文档说明。

#### `TensorBoardXLogger`

TensorBoardX logger implementation.

**参数**

- `log_dir` (`str`): Directory where event files will be written.
- `comment` (`str, default=""`): Comment appended to the log directory name.
- `**kwargs` (`dict`): Additional keyword arguments forwarded to
  ``tensorboardX.SummaryWriter``.

**异常**

- `ImportError`: If ``tensorboardX`` is not installed in the current environment.

##### `TensorBoardXLogger.log_metrics`

```python
log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None
```

未提供文档说明。

##### `TensorBoardXLogger.log_hyperparams`

```python
log_hyperparams(self, params: Dict[str, Any]) -> None
```

未提供文档说明。

##### `TensorBoardXLogger.finish`

```python
finish(self) -> None
```

未提供文档说明。

## `data/`

### `convert`

模块: `torch_rechub.data.convert`

#### `pa_array_to_tensor`

```python
pa_array_to_tensor(arr: pa.Array) -> torch.Tensor
```

Convert a PyArrow array to a PyTorch tensor.

**参数**

- `arr` (`pa.Array`): The given PyArrow array.

**返回**

- `torch.Tensor` (`The result PyTorch tensor.`)

**异常**

- `TypeError`: if the array type or the value type (when nested) is unsupported.
- `ValueError`: if the nested array is ragged (unequal lengths of each row).

### `dataset`

模块: `torch_rechub.data.dataset`

#### `ParquetIterableDataset`

Stream Parquet data as PyTorch tensors.

**参数**

- `file_paths` (`list[FilePath]`): Paths to Parquet files.
- `columns` (`list[str], optional`): Columns to select; if ``None``, read all columns.
- `batch_size` (`int, default _DEFAULT_BATCH_SIZE`): Rows per streamed batch.

**说明**

Reads lazily; no full Parquet load. Each worker gets a partition, builds its
own PyArrow Dataset/Scanner, and yields dicts of column tensors batch by batch.

**示例**

```python
>>> ds = ParquetIterableDataset(
...     ["/data/train1.parquet", "/data/train2.parquet"],
...     columns=["x", "y", "label"],
...     batch_size=1024,
... )
>>> loader = DataLoader(ds, batch_size=None)
>>> for batch in loader:
...     x, y, label = batch["x"], batch["y"], batch["label"]
...     ...
```

## `models/`

### `generative/`

#### `hllm`

模块: `torch_rechub.models.generative.hllm`

##### `HLLMTransformerBlock`

Single HLLM Transformer block with self-attention and FFN.

This block is similar to HSTULayer but designed for HLLM which uses
pre-computed item embeddings as input instead of learnable token embeddings.

**参数**

- `d_model` (`int`): Hidden dimension.
- `n_heads` (`int`): Number of attention heads.
- `dropout` (`float`): Dropout rate.

###### `HLLMTransformerBlock.forward`

```python
forward(self, x, rel_pos_bias = None)
```

Forward pass.

**参数**

- `x` (`Tensor`): Input of shape (B, L, D).
- `rel_pos_bias` (`Tensor, optional`): Relative position bias.

**返回**

- `Tensor` (`Output of shape (B, L, D).`)

##### `HLLMModel`

HLLM: Hierarchical Large Language Model for Recommendation.

This is a lightweight implementation of HLLM that uses pre-computed item
embeddings as input. The original ByteDance HLLM uses end-to-end training
with both Item LLM and User LLM, but this implementation focuses on the
User LLM component for resource efficiency.

Architecture:
    - Item Embeddings: Pre-computed using LLM (offline, frozen)
      Format: "{item_prompt}title: {title}description: {description}"
      where item_prompt = "Compress the following sentence into embedding: "
    - User LLM: Transformer blocks that model user sequences (trainable)
    - Scoring Head: Dot product between user representation and item embeddings

Reference:
    ByteDance HLLM: https://github.com/bytedance/HLLM

**参数**

- `item_embeddings` (`Tensor or str`): Pre-computed item embeddings of shape
  (vocab_size, d_model), or path to a .pt file containing embeddings.
  Generated using the last token's hidden state from an LLM.
- `vocab_size` (`int`): Vocabulary size (number of items).
- `d_model` (`int`): Hidden dimension. Should match item embedding dimension.
  Default: 512. TinyLlama uses 2048, Baichuan2 uses 4096.
- `n_heads` (`int`): Number of attention heads. Default: 8.
- `n_layers` (`int`): Number of transformer blocks. Default: 4.
- `max_seq_len` (`int`): Maximum sequence length. Default: 256.
  Official uses MAX_ITEM_LIST_LENGTH=50.
- `dropout` (`float`): Dropout rate. Default: 0.1.
- `use_rel_pos_bias` (`bool`): Whether to use relative position bias. Default: True.
- `use_time_embedding` (`bool`): Whether to use time embeddings. Default: True.
- `num_time_buckets` (`int`): Number of time buckets. Default: 2048.
- `time_bucket_fn` (`str`): Time bucketization function ('sqrt' or 'log'). Default: 'sqrt'.
- `temperature` (`float`): Temperature for NCE scoring. Default: 1.0.
  Official uses logit_scale = log(1/0.07) ≈ 2.66.

###### `HLLMModel.forward`

```python
forward(self, seq_tokens, time_diffs = None)
```

Forward pass.

**参数**

- `seq_tokens` (`Tensor`): Item token IDs of shape (B, L).
- `time_diffs` (`Tensor, optional`): Time differences in seconds of shape (B, L).

**返回**

- `Tensor` (`Logits of shape (B, L, vocab_size).`)

#### `hstu`

模块: `torch_rechub.models.generative.hstu`

##### `HSTUModel`

HSTU: Hierarchical Sequential Transduction Units.

Autoregressive generative recommender that stacks ``HSTUBlock`` layers to
capture long-range dependencies and predict the next item.

**参数**

- `vocab_size` (`int`): Vocabulary size (items incl. PAD).
- `d_model` (`int, default=512`): Hidden dimension.
- `n_heads` (`int, default=8`): Attention heads.
- `n_layers` (`int, default=4`): Number of stacked HSTU layers.
- `dqk` (`int, default=64`): Query/key dim per head.
- `dv` (`int, default=64`): Value dim per head.
- `max_seq_len` (`int, default=256`): Maximum sequence length.
- `dropout` (`float, default=0.1`): Dropout rate.
- `use_rel_pos_bias` (`bool, default=True`): Use relative position bias.
- `use_time_embedding` (`bool, default=True`): Use time-difference embeddings.
- `num_time_buckets` (`int, default=2048`): Number of time buckets for time embeddings.
- `time_bucket_fn` (`{'sqrt', 'log'}, default='sqrt'`): Bucketization function for time differences.

**张量形状**

- **Input**
  - `x` (```(batch_size, seq_len)```)
  - `time_diffs` (```(batch_size, seq_len)``, optional (seconds).`)
- **Output**
  - `logits` (```(batch_size, seq_len, vocab_size)```)

**示例**

```python
>>> model = HSTUModel(vocab_size=100000, d_model=512)
>>> x = torch.randint(0, 100000, (32, 256))
>>> time_diffs = torch.randint(0, 86400, (32, 256))
>>> logits = model(x, time_diffs)
>>> logits.shape
```
torch.Size([32, 256, 100000])

###### `HSTUModel.forward`

```python
forward(self, x, time_diffs = None)
```

Forward pass.

**参数**

- `x` (`Tensor`): Input token ids of shape ``(batch_size, seq_len)``.
- `time_diffs` (`Tensor, optional`): Time differences in seconds,
  shape ``(batch_size, seq_len)``. If ``None`` and
  ``use_time_embedding=True``, all-zero time differences are used.

**返回**

- `Tensor` (`Logits over the vocabulary of shape`): ``(batch_size, seq_len, vocab_size)``.

#### `rqvae`

模块: `torch_rechub.models.generative.rqvae`

##### `kmeans`

```python
kmeans(samples, num_clusters, num_iters = 10)
```

Perform K-Means clustering on input samples and return cluster centers.

This function applies the scikit-learn implementation of K-Means
to cluster the input samples and returns the resulting cluster
centers as a PyTorch tensor on the original device.

**参数**

- `samples` (`torch.Tensor`): Input tensor of shape (N, D), where N is the number of samples
  and D is the feature dimension.
- `num_clusters` (`int`): The number of clusters to form.
- `num_iters` (`int, optional (default=10)`): Maximum number of iterations of the K-Means algorithm.

**返回**

- `tensor_centers` (`torch.Tensor`): A tensor of shape (num_clusters, D) containing the cluster
  centers, located on the same device as the input samples.

**说明**

This function converts the input tensor to a NumPy array and runs
K-Means on the CPU using scikit-learn. Gradients are not preserved.

##### `sinkhorn_algorithm`

```python
sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations)
```

未提供文档说明。

##### `VectorQuantizer`

VectorQuantizer: Single-stage vector quantization module.

Quantizes input features using a learned codebook and optionally
applies Sinkhorn-based soft assignment. Computes codebook and
commitment losses for training.

**参数**

- `n_e` (`int`): Number of embeddings (codebook size).
- `e_dim` (`int`): Dimensionality of each embedding vector.
- `beta` (`float, default=0.25`): Weight for the commitment loss term.
- `kmeans_init` (`bool, default=False`): Whether to initialize embeddings with K-Means.
- `kmeans_iters` (`int, default=10`): Number of K-Means iterations for initialization.
- `sk_epsilon` (`float, default=0.003`): Entropy regularization coefficient for Sinkhorn assignment.
- `sk_iters` (`int, default=100`): Number of Sinkhorn iterations.

**张量形状**

- **Input**
  - `x` (`torch.Tensor of shape (batch_size, ..., e_dim)`)
- **Output**
  - `x_q` (`torch.Tensor of shape (batch_size, ..., e_dim)`)
  - `loss` (`torch.Tensor, scalar quantization loss`)
  - `indices` (`torch.Tensor of shape (batch_size, ...), codebook indices`)

**示例**

```python
>>> vq = VectorQuantizer(n_e=512, e_dim=64)
>>> x = torch.randn(32, 10, 64)
>>> x_q, loss, indices = vq(x)
>>> x_q.shape
```
torch.Size([32, 10, 64])

###### `VectorQuantizer.get_codebook`

```python
get_codebook(self)
```

Return the current codebook embeddings.

**返回**

- `torch.Tensor`: A tensor of shape (n_e, e_dim) containing the embedding vectors.

###### `VectorQuantizer.get_codebook_entry`

```python
get_codebook_entry(self, indices, shape = None)
```

Retrieve codebook entries corresponding to given indices.

**参数**

- `indices` (`torch.Tensor`): Tensor of indices selecting codebook entries.
- `shape` (`tuple of int, optional`): Desired output shape after reshaping the retrieved embeddings.

**返回**

- `torch.Tensor`: Quantized vectors corresponding to the provided indices.

###### `VectorQuantizer.init_emb`

```python
init_emb(self, data)
```

Initialize the codebook embeddings using K-Means clustering.

###### `VectorQuantizer.center_distance_for_constraint`

```python
center_distance_for_constraint(distances)
```

Center and normalize distance values for constrained optimization.

###### `VectorQuantizer.forward`

```python
forward(self, x, use_sk = True)
```

Apply vector quantization to the input features.

**参数**

- `x` (`torch.Tensor`): Input tensor whose last dimension corresponds to the embedding
  dimension.
- `use_sk` (`bool, optional (default=True)`): Whether to use Sinkhorn-based soft assignment instead of
  hard nearest-neighbor assignment.

**返回**

- `x_q` (`torch.Tensor`): Quantized output tensor with the same shape as the input.
- `loss` (`torch.Tensor`): Vector quantization loss consisting of codebook and commitment
  terms.
- `indices` (`torch.Tensor`): Indices of the selected codebook entries for each input vector.

**说明**

During training, the codebook may be initialized using K-Means
if it has not been initialized yet. Gradients are preserved using
the straight-through estimator.

##### `ResidualVectorQuantizer`

ResidualVectorQuantizer: Multi-stage residual vector quantization.

Applies a sequence of VectorQuantizer modules to progressively
quantize the residuals of the input. Computes mean quantization
loss across all stages. References:SoundStream: An End-to-End Neural Audio Codec
https://arxiv.org/pdf/2107.03312.pdf

**参数**

- `n_e_list` (`list of int`): Number of embeddings for each residual quantization stage.
- `e_dim` (`int`): Dimensionality of each embedding vector.
- `sk_epsilons` (`list of float`): Entropy regularization coefficients for Sinkhorn assignment
  at each stage.
- `beta` (`float, default=0.25`): Weight for the commitment loss term.
- `kmeans_init` (`bool, default=False`): Whether to initialize embeddings with K-Means.
- `kmeans_iters` (`int, default=100`): Number of K-Means iterations for initialization.
- `sk_iters` (`int, default=100`): Number of Sinkhorn iterations.

**张量形状**

- **Input**
  - `x` (`torch.Tensor of shape (batch_size, ..., e_dim)`)
- **Output**
  - `x_q` (`torch.Tensor of shape (batch_size, ..., e_dim)`)
  - `mean_losses` (`torch.Tensor, scalar mean quantization loss`)
  - `all_indices` (`torch.Tensor of shape (batch_size, ..., num_quantizers)`)

**示例**

```python
>>> rvq = ResidualVectorQuantizer(n_e_list=[512, 512], e_dim=64, sk_epsilons=[0.003, 0.003])
>>> x = torch.randn(32, 10, 64)
>>> x_q, loss, indices = rvq(x)
>>> x_q.shape
```
torch.Size([32, 10, 64])

###### `ResidualVectorQuantizer.get_codebook`

```python
get_codebook(self)
```

Return the stacked codebooks from all residual quantizers.

###### `ResidualVectorQuantizer.forward`

```python
forward(self, x, use_sk = True)
```

Apply residual vector quantization to the input features.

**参数**

- `x` (`torch.Tensor`): Input tensor whose last dimension corresponds to the embedding
  dimension.
- `use_sk` (`bool, optional (default=True)`): Whether to use Sinkhorn-based soft assignment for each
  quantization stage.

**返回**

- `x_q` (`torch.Tensor`): Quantized output obtained by summing the outputs of all
  residual quantizers.
- `mean_losses` (`torch.Tensor`): Mean vector quantization loss averaged over all stages.
- `all_indices` (`torch.Tensor`): Tensor containing codebook indices from all quantizers,
  stacked along the last dimension.

**说明**

Each quantization stage operates on the residual from the
previous stage, enabling progressive refinement of the
quantized representation.

##### `RQVAEModel`

RQVAEModel: Residual Quantized Variational Autoencoder.

Implements a VAE with a multi-stage residual vector quantizer
(ResidualVectorQuantizer) for latent discretization.

**参数**

- `in_dim` (`int, default=768`): Input feature dimension.
- `num_emb_list` (`list of int`): Number of embeddings for each residual quantization stage.
- `e_dim` (`int, default=64`): Dimension of each embedding vector.
- `layers` (`list of int`): Hidden layer sizes for the encoder/decoder MLP.
- `dropout_prob` (`float, default=0.0`): Dropout probability applied to MLP layers.
- `bn` (`bool, default=False`): Whether to use batch normalization in MLP layers.
- `loss_type` (`str, default="mse"`): Reconstruction loss type, either "mse" or "l1".
- `quant_loss_weight` (`float, default=1.0`): Weight for the vector quantization loss.
- `beta` (`float, default=0.25`): Commitment loss weight in the vector quantizers.
- `kmeans_init` (`bool, default=False`): Whether to initialize codebooks using K-Means.
- `kmeans_iters` (`int, default=100`): Number of K-Means iterations for initialization.
- `sk_epsilons` (`list of float`): Entropy regularization coefficients for Sinkhorn assignment.
- `sk_iters` (`int, default=100`): Number of Sinkhorn iterations for each quantizer.

**张量形状**

- **Input**
  - `x` (`torch.Tensor of shape (batch_size, in_dim)`)
- **Output**
  - `out` (`torch.Tensor of shape (batch_size, in_dim)`)
  - `rq_loss` (`torch.Tensor, scalar quantization loss`)
  - `indices` (`torch.Tensor of shape (batch_size, num_quantizers)`)

**示例**

```python
>>> model = RQVAEModel(in_dim=768, num_emb_list=[512,512], e_dim=64, layers=[256,128])
>>> x = torch.randn(32, 768)
>>> out, rq_loss, indices = model(x)
>>> out.shape
```
torch.Size([32, 768])

###### `RQVAEModel.forward`

```python
forward(self, x, use_sk = True)
```

Forward pass.

**参数**

- `x` (`torch.Tensor`): Input feature tensor of shape
  ``(batch_size, in_dim)``.
- `use_sk` (`bool, optional`): Whether to use Sinkhorn-based soft
  assignment in the residual vector quantizer. Default: ``True``.

**返回**

- `out` (`torch.Tensor`): Reconstructed output tensor of shape
  ``(batch_size, in_dim)``.
- `rq_loss` (`torch.Tensor`): Scalar residual vector quantization loss.
- `indices` (`torch.Tensor`): Codebook indices from all quantization
  stages, shape ``(batch_size, num_quantizers)``.

###### `RQVAEModel.get_indices`

```python
get_indices(self, xs, use_sk = False)
```

Obtain residual quantizer codebook indices for input features.

**参数**

- `xs` (`torch.Tensor`): Input tensor of shape (batch_size, in_dim)
- `use_sk` (`bool, default=False`): Whether to use Sinkhorn-based soft assignment.

**返回**

- `sids` (`torch.Tensor`)
- `Codebook indices of shape (batch_size, num_quantizers)`

###### `RQVAEModel.compute_loss`

```python
compute_loss(self, out, quant_loss, xs = None)
```

Compute total loss combining reconstruction and quantization losses.

**参数**

- `out` (`torch.Tensor`): Reconstructed output tensor, shape (batch_size, in_dim)
- `quant_loss` (`torch.Tensor`): Vector quantization loss scalar
- `xs` (`torch.Tensor`): Ground-truth input tensor, shape (batch_size, in_dim)

**返回**

- `loss_total` (`torch.Tensor`): Combined reconstruction and quantization loss
- `loss_recon` (`torch.Tensor`): Reconstruction loss only

###### `RQVAEModel.generate_semantic_ids`

```python
generate_semantic_ids(self, data, data_loader, prefix = ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>', '<e_{}>'], use_sk = False, device = 'cuda')
```

Generate semantic IDs for a dataset using the residual vector quantizer.

**参数**

- `data` (`torch.Tensor`): Input dataset of shape (num_samples, in_dim)
- `data_loader` (`torch.utils.data.DataLoader`): DataLoader for iterating over the dataset in batches
- `prefix` (`list of str, default=["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]`): Prefix template for generating semantic ID strings for each quantizer stage
- `use_sk` (`bool, default=False`): Whether to use Sinkhorn-based soft assignment for collisions
- `device` (`str, default='cuda'`): Device to perform computation on

**返回**

- `all_indices_dict` (`dict`): Dictionary mapping item index to list of semantic ID strings from each quantization stage

**示例**

```python
>>> all_indices_dict = model.generate_semantic_ids(data, data_loader)
>>> len(all_indices_dict)
```
num_samples

#### `tiger`

模块: `torch_rechub.models.generative.tiger`

##### `TIGERModel`

未提供文档说明。

###### `TIGERModel.set_hyper`

```python
set_hyper(self, temperature)
```

未提供文档说明。

###### `TIGERModel.ranking_loss`

```python
ranking_loss(self, lm_logits, labels)
```

未提供文档说明。

###### `TIGERModel.forward`

```python
forward(self, input_ids = None, whole_word_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, cross_attn_head_mask = None, past_key_values = None, use_cache = None, labels = None, inputs_embeds = None, decoder_inputs_embeds = None, head_mask = None, decoder_head_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None, reduce_loss = False, return_hidden_state = False, **kwargs)
```



### `matching/`

#### `comirec`

模块: `torch_rechub.models.matching.comirec`

##### `ComirecSA`

The match model mentioned in `Controllable Multi-Interest Framework for Recommendation` paper.
It's a ComirecSA match model trained by global softmax loss on list-wise samples.
Note in origin paper, it's without item dnn tower and train item embedding directly.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `history_features` (`list[Feature Class]`): training history
- `item_features` (`list[Feature Class]`): training by the embedding table, it's the item id feature.
- `neg_item_feature` (`list[Feature Class]`): training by the embedding table, it's the negative items id feature.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

    interest_num （int): interest num

###### `ComirecSA.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `ComirecSA.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `ComirecSA.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

###### `ComirecSA.gen_mask`

```python
gen_mask(self, x)
```

未提供文档说明。

##### `ComirecDR`

The match model mentioned in `Controllable Multi-Interest Framework for Recommendation` paper.
It's a ComirecDR match model trained by global softmax loss on list-wise samples.
Note in origin paper, it's without item dnn tower and train item embedding directly.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `history_features` (`list[Feature Class]`): training history
- `item_features` (`list[Feature Class]`): training by the embedding table, it's the item id feature.
- `neg_item_feature` (`list[Feature Class]`): training by the embedding table, it's the negative items id feature.
- `max_length` (`int`): max sequence length of input item sequence
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

    interest_num （int): interest num

###### `ComirecDR.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `ComirecDR.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `ComirecDR.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

###### `ComirecDR.gen_mask`

```python
gen_mask(self, x)
```

未提供文档说明。

#### `dssm`

模块: `torch_rechub.models.matching.dssm`

##### `DSSM`

Deep Structured Semantic Model

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `item_features` (`list[Feature Class]`): training by the item tower module.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `item_params` (`dict`): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.

###### `DSSM.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `DSSM.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `DSSM.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

#### `dssm_facebook`

模块: `torch_rechub.models.matching.dssm_facebook`

##### `FaceBookDSSM`

Embedding-based Retrieval in Facebook Search
It's a DSSM match model trained by hinge loss on pair-wise samples.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `pos_item_features` (`list[Feature Class]`): negative sample features, training by the item tower module.
- `neg_item_features` (`list[Feature Class]`): positive sample features, training by the item tower module.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `item_params` (`dict`): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.

###### `FaceBookDSSM.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `FaceBookDSSM.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `FaceBookDSSM.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

#### `dssm_senet`

模块: `torch_rechub.models.matching.dssm_senet`

##### `DSSM`

Deep Structured Semantic Model

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `item_features` (`list[Feature Class]`): training by the item tower module.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `item_params` (`dict`): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.

###### `DSSM.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `DSSM.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `DSSM.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

#### `gru4rec`

模块: `torch_rechub.models.matching.gru4rec`

##### `GRU4Rec`

The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
It's a DSSM match model trained by global softmax loss on list-wise samples.
Note in origin paper, it's without item dnn tower and train item embedding directly.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `history_features` (`list[Feature Class]`): training history
- `item_features` (`list[Feature Class]`): training by the embedding table, it's the item id feature.
- `neg_item_feature` (`list[Feature Class]`): training by the embedding table, it's the negative items id feature.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

###### `GRU4Rec.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `GRU4Rec.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `GRU4Rec.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

#### `mind`

模块: `torch_rechub.models.matching.mind`

##### `MIND`

The match model mentioned in `Multi-Interest Network with Dynamic Routing` paper.
It's a ComirecDR match model trained by global softmax loss on list-wise samples.
Note in origin paper, it's without item dnn tower and train item embedding directly.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `history_features` (`list[Feature Class]`): training history
- `item_features` (`list[Feature Class]`): training by the embedding table, it's the item id feature.
- `neg_item_feature` (`list[Feature Class]`): training by the embedding table, it's the negative items id feature.
- `max_length` (`int`): max sequence length of input item sequence
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

    interest_num （int): interest num

###### `MIND.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `MIND.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `MIND.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

###### `MIND.gen_mask`

```python
gen_mask(self, x)
```

未提供文档说明。

#### `narm`

模块: `torch_rechub.models.matching.narm`

##### `NARM`

未提供文档说明。

###### `NARM.user_tower`

```python
user_tower(self, x)
```

Compute user embedding for in-batch negative sampling.

###### `NARM.item_tower`

```python
item_tower(self, x)
```

Compute item embedding for in-batch negative sampling.

###### `NARM.forward`

```python
forward(self, input_dict)
```

未提供文档说明。

#### `sasrec`

模块: `torch_rechub.models.matching.sasrec`

##### `SASRec`

SASRec: Self-Attentive Sequential Recommendation

**参数**

- `features` (`list`): the list of `Feature Class`. In sasrec, the features list needs to have three elements in order: user historical behavior sequence features, positive sample sequence, and negative sample sequence.
- `max_len` (`The length of the sequence feature.`)
- `num_blocks` (`The number of stacks of attention modules.`)
- `num_heads` (`The number of heads in MultiheadAttention.`)
- `item_feature` (`Optional item feature for in-batch negative sampling mode.`)

###### `SASRec.seq_forward`

```python
seq_forward(self, x, embed_x_feature)
```

未提供文档说明。

###### `SASRec.user_tower`

```python
user_tower(self, x)
```

Compute user embedding for in-batch negative sampling.
Takes the last valid position's output as user representation.

###### `SASRec.item_tower`

```python
item_tower(self, x)
```

Compute item embedding for in-batch negative sampling.

###### `SASRec.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `PointWiseFeedForward`

未提供文档说明。

###### `PointWiseFeedForward.forward`

```python
forward(self, inputs)
```

未提供文档说明。

#### `sine`

模块: `torch_rechub.models.matching.sine`

##### `SINE`

The match model was proposed in `Sparse-Interest Network for Sequential Recommendation` paper.

**参数**

- `history_features` (`list[str]`): training history feature names, this is for indexing the historical sequences from input dictionary
- `item_features` (`list[str]`): item feature names, this is for indexing the items from input dictionary
- `neg_item_features` (`list[str]`): neg item feature names, this for indexing negative items from input dictionary
- `num_items` (`int`): number of items in the data
- `embedding_dim` (`int`): dimensionality of the embeddings
- `hidden_dim` (`int`): dimensionality of the hidden layer in self attention modules
- `num_concept` (`int`): number of concept, also called conceptual prototypes
- `num_intention` (`int`): number of (user) specific intentions out of the concepts
- `seq_max_len` (`int`): max sequence length of input item sequence
- `num_heads` (`int`): number of attention heads in self attention modules, default to 1
- `temperature` (`float`): temperature factor in the similarity measure, default to 1.0

###### `SINE.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `SINE.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `SINE.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

###### `SINE.gen_mask`

```python
gen_mask(self, x)
```

未提供文档说明。

#### `stamp`

模块: `torch_rechub.models.matching.stamp`

##### `STAMP`

未提供文档说明。

###### `STAMP.user_tower`

```python
user_tower(self, x)
```

Compute user embedding for in-batch negative sampling.

###### `STAMP.item_tower`

```python
item_tower(self, x)
```

Compute item embedding for in-batch negative sampling.

###### `STAMP.forward`

```python
forward(self, input_dict)
```

未提供文档说明。

#### `youtube_dnn`

模块: `torch_rechub.models.matching.youtube_dnn`

##### `YoutubeDNN`

The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
It's a DSSM match model trained by global softmax loss on list-wise samples.
Note in origin paper, it's without item dnn tower and train item embedding directly.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `item_features` (`list[Feature Class]`): training by the embedding table, it's the item id feature.
- `neg_item_feature` (`list[Feature Class]`): training by the embedding table, it's the negative items id feature.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

###### `YoutubeDNN.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `YoutubeDNN.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `YoutubeDNN.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

#### `youtube_sbc`

模块: `torch_rechub.models.matching.youtube_sbc`

##### `YoutubeSBC`

Sampling-Bias-Corrected Neural Modeling for Matching by Youtube.
It's a DSSM match model trained by In-batch softmax loss on list-wise samples, and add sample debias module.

**参数**

- `user_features` (`list[Feature Class]`): training by the user tower module.
- `item_features` (`list[Feature Class]`): training by the item tower module.
- `sample_weight_feature` (`list[Feature Class]`): used for sampling bias corrected in training.
- `user_params` (`dict`): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `item_params` (`dict`): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `batch_size` (`int`): same as batch size of DataLoader, used in in-batch sampling
- `n_neg` (`int`): the number of negative sample for every positive sample, default to 3. Note it's must smaller than batch_size.
- `temperature` (`float`): temperature factor for similarity score, default to 1.0.

###### `YoutubeSBC.forward`

```python
forward(self, x)
```

未提供文档说明。

###### `YoutubeSBC.user_tower`

```python
user_tower(self, x)
```

未提供文档说明。

###### `YoutubeSBC.item_tower`

```python
item_tower(self, x)
```

未提供文档说明。

### `multi_task/`

#### `aitm`

模块: `torch_rechub.models.multi_task.aitm`

##### `AITM`

Adaptive Information Transfer Multi-task (AITM) framework.
    all the task type must be binary classificatioon.

**参数**

- `features` (`list[Feature Class]`): training by the whole module.
- `n_task` (`int`): the number of binary classificatioon task.
- `bottom_params` (`dict`): the params of all the botwer expert module, keys include:`{"dims":list, "activation":str, "dropout":float}`.
- `tower_params_list` (`list`): the list of tower params dict, the keys same as expert_params.

###### `AITM.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `AttentionLayer`

attention for info tranfer

**参数**

- `dim` (`int`): attention dim

**张量形状**

    Input: (batch_size, 2, dim)
    Output: (batch_size, dim)

###### `AttentionLayer.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `esmm`

模块: `torch_rechub.models.multi_task.esmm`

##### `ESMM`

Entire Space Multi-Task Model

**参数**

- `user_features` (`list`): the list of `Feature Class`, training by shared bottom and tower module. It means the user features.
- `item_features` (`list`): the list of `Feature Class`, training by shared bottom and tower module. It means the item features.
- `cvr_params` (`dict`): the params of the CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
- `ctr_params` (`dict`): the params of the CTR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}

###### `ESMM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `mmoe`

模块: `torch_rechub.models.multi_task.mmoe`

##### `MMOE`

Multi-gate Mixture-of-Experts model.

**参数**

- `features` (`list`): the list of `Feature Class`, training by the expert and tower module.
- `task_types` (`list`): types of tasks, only support `["classfication", "regression"]`.
- `n_expert` (`int`): the number of expert net.
- `expert_params` (`dict`): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
- `tower_params_list` (`list`): the list of tower params dict, the keys same as expert_params.

###### `MMOE.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `ple`

模块: `torch_rechub.models.multi_task.ple`

##### `PLE`

Progressive Layered Extraction model.

**参数**

- `features` (`list`): the list of `Feature Class`, training by the expert and tower module.
- `task_types` (`list`): types of tasks, only support `["classfication", "regression"]`.
- `n_level` (`int`): the  number of CGC layer.
- `n_expert_specific` (`int`): the number of task-specific expert net.
- `n_expert_shared` (`int`): the number of task-shared expert net.
- `expert_params` (`dict`): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
- `tower_params_list` (`list`): the list of tower params dict, the keys same as expert_params.

###### `PLE.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `CGC`

Customized Gate Control (CGC) Model mentioned in PLE paper.

**参数**

- `cur_level` (`int`): the current level of CGC in PLE.
- `n_level` (`int`): the  number of CGC layer.
- `n_task` (`int`): the number of tasks.
- `n_expert_specific` (`int`): the number of task-specific expert net.
- `n_expert_shared` (`int`): the number of task-shared expert net.
- `input_dims` (`int`): the input dims of the xpert module in current CGC layer.
- `expert_params` (`dict`): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.

###### `CGC.forward`

```python
forward(self, x_list)
```

未提供文档说明。

#### `shared_bottom`

模块: `torch_rechub.models.multi_task.shared_bottom`

##### `SharedBottom`

Shared Bottom multi task model.

**参数**

- `features` (`list`): the list of `Feature Class`, training by the bottom and tower module.
- `task_types` (`list`): types of tasks, only support `["classfication", "regression"]`.
- `bottom_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float}, keep `{"output_layer":False}`.
- `tower_params_list` (`list`): the list of tower params dict, the keys same as bottom_params.

###### `SharedBottom.forward`

```python
forward(self, x)
```

未提供文档说明。

### `ranking/`

#### `afm`

模块: `torch_rechub.models.ranking.afm`

##### `AFM`

Attentional Factorization Machine Model

**参数**

- `fm_features` (`list`): the list of `Feature Class`, training by the fm part module.
- `embed_dim` (`int`): the dimension of input embedding.
- `t` (`int`): the size of the hidden layer in the attention network.

###### `AFM.attention`

```python
attention(self, y_fm)
```

未提供文档说明。

###### `AFM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `autoint`

模块: `torch_rechub.models.ranking.autoint`

##### `AutoInt`

AutoInt Model

**参数**

- `sparse_features` (`list`): the list of `SparseFeature` Class
- `dense_features` (`list`): the list of `DenseFeature` Class
- `num_layers` (`int`): number of interacting layers
- `num_heads` (`int`): number of attention heads
- `dropout` (`float`): dropout rate for attention
- `mlp_params` (`dict`): parameters for MLP, keys: {"dims":list, "activation":str,
  "dropout":float, "output_layer":bool"}

###### `AutoInt.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `bst`

模块: `torch_rechub.models.ranking.bst`

##### `BST`

Behavior Sequence Transformer

**参数**

- `features` (`list`): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
- `history_features` (`list`): the list of `Feature Class`,training by Transformer. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
- `target_features` (`list`): the list of `Feature Class`, training by Transformer. It means the target feature which will execute target-attention with history feature.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
- `nhead` (`int`): the number of heads in the multi-head-attention models.
- `dropout` (`float`): the dropout value in the multi-head-attention models.
- `num_layers` (`Any`): the number of sub-encoder-layers in the encoder.

###### `BST.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `dcn`

模块: `torch_rechub.models.ranking.dcn`

##### `DCN`

Deep & Cross Network

**参数**

- `features` (`list[Feature Class]`): training by the whole module.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}

###### `DCN.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `dcn_v2`

模块: `torch_rechub.models.ranking.dcn_v2`

##### `DCNv2`

Deep & Cross Network with a mixture of low-rank architecture

**参数**

- `features` (`list[Feature Class]`): training by the whole module.
- `n_cross_layers` (`int`): the number of layers of feature intersection layers
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
- `use_low_rank_mixture` (`bool`): True, whether to use a mixture of low-rank architecture
- `low_rank` (`int`): the rank size of low-rank matrices
- `num_experts` (`int`): the number of expert networks

###### `DCNv2.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `deepffm`

模块: `torch_rechub.models.ranking.deepffm`

##### `DeepFFM`

The DeepFFM model, mentioned on the `webpage
<https://cs.nju.edu.cn/31/60/c1654a209248/page.htm>` which is the first
work that introduces FFM model into neural CTR system. It is also described
in the `FAT-DeepFFM paper <https://arxiv.org/abs/1905.06336>`.

**参数**

- `linear_features` (`list`): the list of `Feature Class`, fed to the linear module.
- `cross_features` (`list`): the list of `Feature Class`, fed to the ffm module.
- `embed_dim` (`int`): the dimensionality of categorical value embedding.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}

###### `DeepFFM.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `FatDeepFFM`

The FAT-DeepFFM model, mentioned in the `FAT-DeepFFM paper
<https://arxiv.org/abs/1905.06336>`. It combines DeepFFM with
Compose-Excitation Network (CENet) field attention mechanism
to highlight the importance of second-order feature crosses.

**参数**

- `linear_features` (`list`): the list of `Feature Class`, fed to the linear module.
- `cross_features` (`list`): the list of `Feature Class`, fed to the ffm module.
- `embed_dim` (`int`): the dimensionality of categorical value embedding.
- `reduction_ratio` (`int`): the between the dimensions of input layer and hidden layer of the CEN MLP module.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}

###### `FatDeepFFM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `deepfm`

模块: `torch_rechub.models.ranking.deepfm`

##### `DeepFM`

Deep Factorization Machine Model

**参数**

- `deep_features` (`list`): the list of `Feature Class`, training by the deep part module.
- `fm_features` (`list`): the list of `Feature Class`, training by the fm part module.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}

###### `DeepFM.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `dien`

模块: `torch_rechub.models.ranking.dien`

##### `AUGRU`

未提供文档说明。

###### `AUGRU.forward`

```python
forward(self, x, item)
```

:param x: 输入的序列向量，维度为 [ batch_size, seq_lens, embed_dim ]
:param item: 目标物品的向量
:return: outs: 所有AUGRU单元输出的隐藏向量[ batch_size, seq_lens, embed_dim ]
         h: 最后一个AUGRU单元输出的隐藏向量[ batch_size, embed_dim ]

##### `AUGRU_Cell`

未提供文档说明。

###### `AUGRU_Cell.attention`

```python
attention(self, x, item)
```

:param x: 输入的序列中第t个向量 [ batch_size, embed_dim ]
:param item: 目标物品的向量 [ batch_size, embed_dim ]
:return: 注意力权重 [ batch_size, 1 ]

###### `AUGRU_Cell.forward`

```python
forward(self, x, h_1, item)
```

:param x:  输入的序列中第t个物品向量 [ batch_size, embed_dim ]
:param h_1:  上一个AUGRU单元输出的隐藏向量 [ batch_size, embed_dim ]
:param item: 目标物品的向量 [ batch_size, embed_dim ]
:return: h 当前层输出的隐藏向量 [ batch_size, embed_dim ]

##### `DIEN`

Deep Interest Evolution Network

**参数**

- `features` (`list`): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
- `history_features` (`list`): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
- `target_features` (`list`): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
- `history_labels` (`list`): the list of history_features whether it is clicked history or not. It should be 0 or 1.
- `alpha` (`float`): the weighting of auxiliary loss.

###### `DIEN.auxiliary`

```python
auxiliary(self, outs, history_features, history_labels)
```

:param history_features: 历史序列物品的向量 [ batch_size, len_seqs, dim ]
:param outs: 兴趣抽取层GRU网络输出的outs [ batch_size, len_seqs, dim ]
:param history_labels: 历史序列物品标注 [ batch_size, len_seqs, 1 ]
:return: 辅助损失函数

###### `DIEN.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `din`

模块: `torch_rechub.models.ranking.din`

##### `DIN`

Deep Interest Network

**参数**

- `features` (`list`): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
- `history_features` (`list`): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
- `target_features` (`list`): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
- `attention_mlp_params` (`dict`): the params of the ActivationUnit module, keys include:`{"dims":list, "activation":str, "dropout":float, "use_softmax":bool`}

###### `DIN.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `ActivationUnit`

Activation Unit Layer mentioned in DIN paper, it is a Target Attention method.

**参数**

- `embed_dim` (`int`): the length of embedding vector.
- `history` (`tensor`)

**张量形状**

    - Input: `(batch_size, seq_length, emb_dim)`
    - Output: `(batch_size, emb_dim)`

###### `ActivationUnit.forward`

```python
forward(self, history, target)
```

未提供文档说明。

#### `edcn`

模块: `torch_rechub.models.ranking.edcn`

##### `EDCN`

Deep & Cross Network with a mixture of low-rank architecture

**参数**

- `features` (`list[Feature Class]`): training by the whole module.
- `n_cross_layers` (`int`): the number of layers of feature intersection layers
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
- `bridge_type` (`str`): the type interaction function, in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"]
- `use_regulation_module` (`bool`): True, whether to use regulation module
- `temperature` (`int`): the temperature coefficient to control distribution

###### `EDCN.forward`

```python
forward(self, x)
```

未提供文档说明。

##### `BridgeModule`

未提供文档说明。

###### `BridgeModule.forward`

```python
forward(self, x, h)
```

未提供文档说明。

##### `RegulationModule`

未提供文档说明。

###### `RegulationModule.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `fibinet`

模块: `torch_rechub.models.ranking.fibinet`

##### `FiBiNet`

**参数**

- `features` (`list[Feature Class]`): training by the whole module.
- `reduction_ratio` (`int`): Hidden layer reduction factor of SENET layer
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
- `bilinear_type` (`str`): the type bilinear interaction function, in ["field_all", "field_each", "field_interaction"], field_all means that all features share a W, field_each means that a feature field corresponds to a W_i, field_interaction means that a feature field intersection corresponds to a W_ij

###### `FiBiNet.forward`

```python
forward(self, x)
```

未提供文档说明。

#### `widedeep`

模块: `torch_rechub.models.ranking.widedeep`

##### `WideDeep`

Wide & Deep Learning model.

**参数**

- `wide_features` (`list`): the list of `Feature Class`, training by the wide part module.
- `deep_features` (`list`): the list of `Feature Class`, training by the deep part module.
- `mlp_params` (`dict`): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}

###### `WideDeep.forward`

```python
forward(self, x)
```

未提供文档说明。

## `serving/`

### `annoy`

模块: `torch_rechub.serving.annoy`

#### `AnnoyBuilder`

ANNOY-based implementation of ``BaseBuilder``.

##### `AnnoyBuilder.from_embeddings`

```python
from_embeddings(self, embeddings: torch.Tensor) -> ty.Generator['AnnoyIndexer', None, None]
```

Adhere to ``BaseBuilder.from_embeddings``.

##### `AnnoyBuilder.from_index_file`

```python
from_index_file(self, index_file: FilePath) -> ty.Generator['AnnoyIndexer', None, None]
```

Adhere to ``BaseBuilder.from_index_file``.

#### `AnnoyIndexer`

ANNOY-based implementation of ``BaseIndexer``.

##### `AnnoyIndexer.query`

```python
query(self, embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]
```

Adhere to ``BaseIndexer.query``.

##### `AnnoyIndexer.save`

```python
save(self, file_path: FilePath) -> None
```

Adhere to ``BaseIndexer.save``.

### `base`

模块: `torch_rechub.serving.base`

#### `BaseBuilder`

Abstract base class for vector index construction.

A builder owns all build-time configuration and produces a ``BaseIndexer`` through a
context-managed build operation.

**示例**

```python
>>> builder = BaseBuilder(...)
>>> embeddings = torch.randn(1000, 128)
>>> with builder.from_embeddings(embeddings) as indexer:
...     ids, scores = indexer.query(embeddings[:2], top_k=5)
...     indexer.save("index.bin")
>>> with builder.from_index_file("index.bin") as indexer:
...     ids, scores = indexer.query(embeddings[:2], top_k=5)
```

##### `BaseBuilder.from_embeddings`

```python
from_embeddings(self, embeddings: torch.Tensor) -> ty.ContextManager['BaseIndexer']
```

Build a vector index from the embeddings.

**参数**

- `embeddings` (`torch.Tensor`): A 2D tensor (n, d) containing embedding vectors to build a new index.

**返回**

- `ContextManager[BaseIndexer]`: A context manager that yields a fully initialized ``BaseIndexer``.

##### `BaseBuilder.from_index_file`

```python
from_index_file(self, index_file: FilePath) -> ty.ContextManager['BaseIndexer']
```

Build a vector index from the index file.

**参数**

- `index_file` (`FilePath`): Path to a serialized index on disk to be loaded.

**返回**

- `ContextManager[BaseIndexer]`: A context manager that yields a fully initialized ``BaseIndexer``.

#### `BaseIndexer`

Abstract base class for vector indexers in the retrieval stage.

##### `BaseIndexer.query`

```python
query(self, embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]
```

Query the vector index.

**参数**

- `embeddings` (`torch.Tensor`): A 2D tensor (n, d) containing embedding vectors to query the index.
- `top_k` (`int`): The number of nearest items to retrieve for each vector.

**返回**

- `torch.Tensor`: A 2D tensor of shape (n, top_k), containing the retrieved nearest neighbor
  IDs for each vector, ordered by descending relevance.
- `torch.Tensor`: A 2D tensor of shape (n, top_k), containing the relevance distances of the
  nearest neighbors for each vector.

##### `BaseIndexer.save`

```python
save(self, file_path: FilePath) -> None
```

Persist the index to local disk.

**参数**

- `file_path` (`FilePath`): Destination path where the index will be saved.

### `faiss`

模块: `torch_rechub.serving.faiss`

#### `FaissBuilder`

Implement ``BaseBuilder`` for FAISS vector index construction.

##### `FaissBuilder.from_embeddings`

```python
from_embeddings(self, embeddings: torch.Tensor) -> ty.Generator['FaissIndexer', None, None]
```

Adhere to ``BaseBuilder.from_embeddings``.

##### `FaissBuilder.from_index_file`

```python
from_index_file(self, index_file: FilePath) -> ty.Generator['FaissIndexer', None, None]
```

Adhere to ``BaseBuilder.from_index_file``.

#### `FaissIndexer`

FAISS-based implementation of ``BaseIndexer``.

##### `FaissIndexer.query`

```python
query(self, embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]
```

Adhere to ``BaseIndexer.query``.

##### `FaissIndexer.save`

```python
save(self, file_path: FilePath) -> None
```

Adhere to ``BaseIndexer.save``.

### `milvus`

模块: `torch_rechub.serving.milvus`

#### `MilvusBuilder`

Implement ``BaseBuilder`` for Milvus vector index construction.

##### `MilvusBuilder.from_embeddings`

```python
from_embeddings(self, embeddings: torch.Tensor) -> ty.Generator['MilvusIndexer', None, None]
```

Adhere to ``BaseBuilder.from_embeddings``.

##### `MilvusBuilder.from_index_file`

```python
from_index_file(self, index_file: FilePath) -> ty.Generator['MilvusIndexer', None, None]
```

Adhere to ``BaseBuilder.from_index_file``.

#### `MilvusIndexer`

Milvus-based implementation of ``BaseIndexer``.

##### `MilvusIndexer.query`

```python
query(self, embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]
```

Adhere to ``BaseIndexer.query``.

##### `MilvusIndexer.save`

```python
save(self, file_path: FilePath) -> None
```

Adhere to ``BaseIndexer.save``.

## `trainers/`

### `ctr_trainer`

模块: `torch_rechub.trainers.ctr_trainer`

#### `CTRTrainer`

A general trainer for single task learning.

**参数**

- `model` (`nn.Module`): any multi task learning model.
- `optimizer_fn` (`torch.optim`): optimizer function of pytorch (default = `torch.optim.Adam`).
- `optimizer_params` (`dict`): parameters of optimizer_fn.
- `scheduler_fn` (`torch.optim.lr_scheduler`): torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
- `scheduler_params` (`dict`): parameters of optimizer scheduler_fn.
- `n_epoch` (`int`): epoch number of training.
- `earlystop_patience` (`int`): how long to wait after last time validation auc improved (default=10).
- `device` (`str`): `"cpu"` or `"cuda:0"`
- `gpus` (`list`): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
- `loss_mode` (`bool`): whether the model returns only prediction or prediction with extra loss
  (`True`: `model(x_dict) -> y_pred`, `False`: `model(x_dict) -> (y_pred, other_loss)`).
- `model_path` (`str`): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
- `embedding_l1` (`float`): L1 regularization coefficient for embedding parameters (default=0.0).
- `embedding_l2` (`float`): L2 regularization coefficient for embedding parameters (default=0.0).
- `dense_l1` (`float`): L1 regularization coefficient for dense parameters (default=0.0).
- `dense_l2` (`float`): L2 regularization coefficient for dense parameters (default=0.0).

##### `CTRTrainer.train_one_epoch`

```python
train_one_epoch(self, data_loader, log_interval = 10)
```

未提供文档说明。

##### `CTRTrainer.fit`

```python
fit(self, train_dataloader, val_dataloader = None)
```

未提供文档说明。

##### `CTRTrainer.evaluate`

```python
evaluate(self, model, data_loader)
```

未提供文档说明。

##### `CTRTrainer.predict`

```python
predict(self, model, data_loader)
```

未提供文档说明。

##### `CTRTrainer.export_onnx`

```python
export_onnx(self, output_path, dummy_input = None, batch_size = 2, seq_length = 10, opset_version = 14, dynamic_batch = True, device = None, verbose = False, onnx_export_kwargs = None)
```

Export the trained model to ONNX format.

This method exports the ranking model (e.g., DeepFM, WideDeep, DCN) to ONNX format
for deployment. The export is non-invasive and does not modify the model code.

**参数**

- `output_path` (`str`): Path to save the ONNX model file.
- `dummy_input` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int`): Batch size for auto-generated dummy input (default: 2).
- `seq_length` (`int`): Sequence length for SequenceFeature (default: 10).
- `opset_version` (`int`): ONNX opset version (default: 14).
- `dynamic_batch` (`bool`): Enable dynamic batch size (default: True).
- `device` (`str, optional`): Device for export ('cpu', 'cuda', etc.).
  If None, defaults to 'cpu' for maximum compatibility.
- `verbose` (`bool`): Print export details (default: False).
- `onnx_export_kwargs` (`dict, optional`): Extra kwargs forwarded to ``torch.onnx.export``.
  The exporter tries the dynamo path first and falls back to the
  legacy exporter automatically. Pass ``{"dynamo": False}`` here
  to force legacy export when needed.

**返回**

- `bool` (`True if export succeeded, False otherwise.`)

**示例**

```python
>>> trainer = CTRTrainer(model, ...)
>>> trainer.fit(train_dl, val_dl)
>>> trainer.export_onnx("deepfm.onnx")

>>> # With custom dummy input
>>> dummy = {"user_id": torch.tensor([1, 2]), "item_id": torch.tensor([10, 20])}
>>> trainer.export_onnx("model.onnx", dummy_input=dummy)

>>> # Export on specific device
>>> trainer.export_onnx("model.onnx", device="cpu")
```

##### `CTRTrainer.visualization`

```python
visualization(self, input_data = None, batch_size = 2, seq_length = 10, depth = 3, show_shapes = True, expand_nested = True, save_path = None, graph_name = 'model', device = None, dpi = 300, **kwargs)
```

Visualize the model's computation graph.

This method generates a visual representation of the model architecture,
showing layer connections, tensor shapes, and nested module structures.
It automatically extracts feature information from the model.

**参数**

- `input_data` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int`): Batch size for auto-generated dummy input (default: 2).
- `seq_length` (`int`): Sequence length for SequenceFeature (default: 10).
- `depth` (`int`): Visualization depth, higher values show more detail.
  Set to -1 to show all layers (default: 3).
- `show_shapes` (`bool`): Whether to display tensor shapes (default: True).
- `expand_nested` (`bool`): Whether to expand nested modules (default: True).
- `save_path` (`str, optional`): Path to save the graph image (.pdf, .svg, .png).
  If None, displays in Jupyter or opens system viewer.
- `graph_name` (`str`): Name for the graph (default: "model").
- `device` (`str, optional`): Device for model execution. If None, defaults to 'cpu'.
- `dpi` (`int`): Resolution in dots per inch for output image.
  Higher values produce sharper images suitable for papers (default: 300).
- `**kwargs` (`Additional arguments passed to ``torchview.draw_graph()``.`)

**返回**

- `ComputationGraph` (`A torchview ComputationGraph object.`)

**异常**

- `ImportError` (`If torchview or graphviz is not installed.`)

**说明**

    When ``save_path`` is None (default):
    - In Jupyter/IPython: automatically displays the graph inline
    - In Python script: opens the graph with system default viewer

**示例**

```python
>>> trainer = CTRTrainer(model, ...)
>>> trainer.fit(train_dl, val_dl)
>>>
>>> # Auto-display in Jupyter (no save_path needed)
>>> trainer.visualization(depth=4)
>>>
>>> # Save to high-DPI PNG for papers
>>> trainer.visualization(save_path="model.png", dpi=300)
```

### `match_trainer`

模块: `torch_rechub.trainers.match_trainer`

#### `MatchTrainer`

A general trainer for Matching/Retrieval

**参数**

- `model` (`nn.Module`): any matching model.
- `mode` (`int, optional`): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
- `optimizer_fn` (`torch.optim`): optimizer function of pytorch (default = `torch.optim.Adam`).
- `optimizer_params` (`dict`): parameters of optimizer_fn.
- `scheduler_fn` (`torch.optim.lr_scheduler`): torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
- `scheduler_params` (`dict`): parameters of optimizer scheduler_fn.
- `n_epoch` (`int`): epoch number of training.
- `earlystop_patience` (`int`): how long to wait after last time validation auc improved (default=10).
- `device` (`str`): `"cpu"` or `"cuda:0"`
- `gpus` (`list`): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
- `model_path` (`str`): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
- `in_batch_neg` (`bool`): whether to use in-batch negative sampling instead of global negatives.
- `in_batch_neg_ratio` (`int`): number of negatives to draw from the batch per positive sample when in_batch_neg is True.
- `hard_negative` (`bool`): whether to choose hardest negatives within batch (top-k by score) instead of uniform random.
- `sampler_seed` (`int`): optional random seed for in-batch sampler to ease reproducibility/testing.

##### `MatchTrainer.train_one_epoch`

```python
train_one_epoch(self, data_loader, log_interval = 10)
```

未提供文档说明。

##### `MatchTrainer.fit`

```python
fit(self, train_dataloader, val_dataloader = None)
```

未提供文档说明。

##### `MatchTrainer.evaluate`

```python
evaluate(self, model, data_loader)
```

未提供文档说明。

##### `MatchTrainer.predict`

```python
predict(self, model, data_loader)
```

未提供文档说明。

##### `MatchTrainer.inference_embedding`

```python
inference_embedding(self, model, mode, data_loader, model_path)
```

未提供文档说明。

##### `MatchTrainer.export_onnx`

```python
export_onnx(self, output_path, mode = None, dummy_input = None, batch_size = 2, seq_length = 10, opset_version = 14, dynamic_batch = True, device = None, verbose = False, onnx_export_kwargs = None)
```

Export the trained matching model to ONNX format.

This method exports matching/retrieval models (e.g., DSSM, YoutubeDNN, MIND)
to ONNX format. For dual-tower models, you can export user tower and item
tower separately for efficient online serving.

**参数**

- `output_path` (`str`): Path to save the ONNX model file.
- `mode` (`str, optional`): Export mode for dual-tower models:
  - "user": Export only the user tower (for user embedding inference)
  - "item": Export only the item tower (for item embedding inference)
  - None: Export the full model (default)
- `dummy_input` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int`): Batch size for auto-generated dummy input (default: 2).
- `seq_length` (`int`): Sequence length for SequenceFeature (default: 10).
- `opset_version` (`int`): ONNX opset version (default: 14).
- `dynamic_batch` (`bool`): Enable dynamic batch size (default: True).
- `device` (`str, optional`): Device for export ('cpu', 'cuda', etc.).
  If None, defaults to 'cpu' for maximum compatibility.
- `verbose` (`bool`): Print export details (default: False).
- `onnx_export_kwargs` (`dict, optional`): Extra kwargs forwarded to ``torch.onnx.export``.
  The exporter tries the dynamo path first and falls back to the
  legacy exporter automatically. Pass ``{"dynamo": False}`` here
  to force legacy export when needed.

**返回**

- `bool` (`True if export succeeded, False otherwise.`)

**示例**

```python
>>> trainer = MatchTrainer(dssm_model, mode=0, ...)
>>> trainer.fit(train_dl)

>>> # Export user tower for user embedding inference
>>> trainer.export_onnx("user_tower.onnx", mode="user")

>>> # Export item tower for item embedding inference
>>> trainer.export_onnx("item_tower.onnx", mode="item")

>>> # Export full model (for online similarity computation)
>>> trainer.export_onnx("full_model.onnx")

>>> # Export on specific device
>>> trainer.export_onnx("user_tower.onnx", mode="user", device="cpu")
```

##### `MatchTrainer.visualization`

```python
visualization(self, input_data = None, batch_size = 2, seq_length = 10, depth = 3, show_shapes = True, expand_nested = True, save_path = None, graph_name = 'model', device = None, dpi = 300, **kwargs)
```

Visualize the model's computation graph.

This method generates a visual representation of the model architecture,
showing layer connections, tensor shapes, and nested module structures.
It automatically extracts feature information from the model.

**参数**

- `input_data` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int, default=2`): Batch size for auto-generated dummy input.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature.
- `depth` (`int, default=3`): Visualization depth, higher values show more detail.
  Set to -1 to show all layers.
- `show_shapes` (`bool, default=True`): Whether to display tensor shapes.
- `expand_nested` (`bool, default=True`): Whether to expand nested modules.
- `save_path` (`str, optional`): Path to save the graph image (.pdf, .svg, .png).
  If None, displays in Jupyter or opens system viewer.
- `graph_name` (`str, default="model"`): Name for the graph.
- `device` (`str, optional`): Device for model execution. If None, defaults to 'cpu'.
- `dpi` (`int, default=300`): Resolution in dots per inch for output image.
  Higher values produce sharper images suitable for papers.
- `**kwargs` (`dict`): Additional arguments passed to torchview.draw_graph().

**返回**

- `ComputationGraph`: A torchview ComputationGraph object.

**异常**

- `ImportError`: If torchview or graphviz is not installed.

**说明**

Default Display Behavior:
    When `save_path` is None (default):
    - In Jupyter/IPython: automatically displays the graph inline
    - In Python script: opens the graph with system default viewer

**示例**

```python
>>> trainer = MatchTrainer(model, ...)
>>> trainer.fit(train_dl)
>>>
>>> # Auto-display in Jupyter (no save_path needed)
>>> trainer.visualization(depth=4)
>>>
>>> # Save to high-DPI PNG for papers
>>> trainer.visualization(save_path="model.png", dpi=300)
```

### `mtl_trainer`

模块: `torch_rechub.trainers.mtl_trainer`

#### `MTLTrainer`

A trainer for multi task learning.

**参数**

- `model` (`nn.Module`): any multi task learning model.
- `task_types` (`list`): types of tasks, only support ["classfication", "regression"].
- `optimizer_fn` (`torch.optim`): optimizer function of pytorch (default = `torch.optim.Adam`).
- `optimizer_params` (`dict`): parameters of optimizer_fn.
- `scheduler_fn` (`torch.optim.lr_scheduler`): torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
- `scheduler_params` (`dict`): parameters of optimizer scheduler_fn.
- `adaptive_params` (`dict`): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`.
- `n_epoch` (`int`): epoch number of training.
- `earlystop_taskid` (`int`): task id of earlystop metrics relies between multi task (default = 0).
- `earlystop_patience` (`int`): how long to wait after last time validation auc improved (default = 10).
- `device` (`str`): `"cpu"` or `"cuda:0"`
- `gpus` (`list`): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
- `model_path` (`str`): the path you want to save the model (default="./"). Note only save the best weight in the validation data.

##### `MTLTrainer.train_one_epoch`

```python
train_one_epoch(self, data_loader)
```

未提供文档说明。

##### `MTLTrainer.fit`

```python
fit(self, train_dataloader, val_dataloader, mode = 'base', seed = 0)
```

未提供文档说明。

##### `MTLTrainer.evaluate`

```python
evaluate(self, model, data_loader)
```

未提供文档说明。

##### `MTLTrainer.predict`

```python
predict(self, model, data_loader)
```

未提供文档说明。

##### `MTLTrainer.export_onnx`

```python
export_onnx(self, output_path, dummy_input = None, batch_size = 2, seq_length = 10, opset_version = 14, dynamic_batch = True, device = None, verbose = False, onnx_export_kwargs = None)
```

Export the trained multi-task model to ONNX format.

This method exports multi-task learning models (e.g., MMOE, PLE, ESMM, SharedBottom)
to ONNX format for deployment. The exported model will have multiple outputs
corresponding to each task.

**说明**

    The ONNX model will output a tensor of shape [batch_size, n_task] where
    n_task is the number of tasks in the multi-task model.

**参数**

- `output_path` (`str`): Path to save the ONNX model file.
- `dummy_input` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int`): Batch size for auto-generated dummy input (default: 2).
- `seq_length` (`int`): Sequence length for SequenceFeature (default: 10).
- `opset_version` (`int`): ONNX opset version (default: 14).
- `dynamic_batch` (`bool`): Enable dynamic batch size (default: True).
- `device` (`str, optional`): Device for export ('cpu', 'cuda', etc.).
  If None, defaults to 'cpu' for maximum compatibility.
- `verbose` (`bool`): Print export details (default: False).
- `onnx_export_kwargs` (`dict, optional`): Extra kwargs forwarded to ``torch.onnx.export``.
  The exporter tries the dynamo path first and falls back to the
  legacy exporter automatically. Pass ``{"dynamo": False}`` here
  to force legacy export when needed.

**返回**

- `bool` (`True if export succeeded, False otherwise.`)

**示例**

```python
>>> trainer = MTLTrainer(mmoe_model, task_types=["classification", "classification"], ...)
>>> trainer.fit(train_dl, val_dl)
>>> trainer.export_onnx("mmoe.onnx")

>>> # Export on specific device
>>> trainer.export_onnx("mmoe.onnx", device="cpu")
```

##### `MTLTrainer.visualization`

```python
visualization(self, input_data = None, batch_size = 2, seq_length = 10, depth = 3, show_shapes = True, expand_nested = True, save_path = None, graph_name = 'model', device = None, dpi = 300, **kwargs)
```

Visualize the model's computation graph.

This method generates a visual representation of the model architecture,
showing layer connections, tensor shapes, and nested module structures.
It automatically extracts feature information from the model.

**参数**

- `input_data` (`dict, optional`): Example input dict {feature_name: tensor}.
  If not provided, dummy inputs will be generated automatically.
- `batch_size` (`int, default=2`): Batch size for auto-generated dummy input.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature.
- `depth` (`int, default=3`): Visualization depth, higher values show more detail.
  Set to -1 to show all layers.
- `show_shapes` (`bool, default=True`): Whether to display tensor shapes.
- `expand_nested` (`bool, default=True`): Whether to expand nested modules.
- `save_path` (`str, optional`): Path to save the graph image (.pdf, .svg, .png).
  If None, displays in Jupyter or opens system viewer.
- `graph_name` (`str, default="model"`): Name for the graph.
- `device` (`str, optional`): Device for model execution. If None, defaults to 'cpu'.
- `dpi` (`int, default=300`): Resolution in dots per inch for output image.
  Higher values produce sharper images suitable for papers.
- `**kwargs` (`dict`): Additional arguments passed to torchview.draw_graph().

**返回**

- `ComputationGraph`: A torchview ComputationGraph object.

**异常**

- `ImportError`: If torchview or graphviz is not installed.

**说明**

Default Display Behavior:
    When `save_path` is None (default):
    - In Jupyter/IPython: automatically displays the graph inline
    - In Python script: opens the graph with system default viewer

**示例**

```python
>>> trainer = MTLTrainer(model, task_types=["classification", "classification"])
>>> trainer.fit(train_dl, val_dl)
>>>
>>> # Auto-display in Jupyter (no save_path needed)
>>> trainer.visualization(depth=4)
>>>
>>> # Save to high-DPI PNG for papers
>>> trainer.visualization(save_path="model.png", dpi=300)
```

### `rqvae_trainer`

模块: `torch_rechub.trainers.rqvae_trainer`

#### `Trainer`

Training utility class for PyTorch models.

Handles the full training loop including optimization, evaluation,
checkpointing, and logging.

**参数**

- `model` (`torch.nn.Module`): Model to be trained.
- `optimizer_fn` (`callable, default=torch.optim.Adam`): Optimizer constructor.
- `optimizer_params` (`dict, optional`): Parameters passed to the optimizer.
- `scheduler_fn` (`callable, optional`): Learning rate scheduler constructor.
- `scheduler_params` (`dict, optional`): Parameters passed to the scheduler.
- `n_epoch` (`int, default=10`): Number of training epochs.
- `device` (`str, default='cpu'`): Device used for training.
- `model_path` (`str, default='./'`): Directory to save model checkpoints.
- `model_logger` (`object or list, optional`): Logger instance(s) used for recording metrics.
- `eval_step` (`int, default=50`): Evaluation interval measured in epochs.

**属性**

- `best_loss` (`float`): Best training loss observed so far.
- `best_collision_rate` (`float`): Best collision rate observed during evaluation.

##### `Trainer.train_one_epoch`

```python
train_one_epoch(self, data_loader)
```

Train the model for a single epoch.

**参数**

- `data_loader` (`torch.utils.data.DataLoader`): DataLoader providing training batches.

**返回**

- `total_loss` (`float`): Sum of total training loss over the epoch.
- `total_recon_loss` (`float`): Sum of reconstruction loss over the epoch.

##### `Trainer.evaluate`

```python
evaluate(self, data_loader)
```

Evaluate the model by computing collision rate.

**参数**

- `data_loader` (`torch.utils.data.DataLoader`): DataLoader providing evaluation data.

**返回**

- `collision_rate` (`float`): Ratio of duplicate semantic codes among all samples.

##### `Trainer.fit`

```python
fit(self, train_dataloader)
```

Run the full training procedure.

Performs iterative training, periodic evaluation, metric logging,
and checkpoint saving.

**参数**

- `train_dataloader` (`torch.utils.data.DataLoader`): DataLoader providing training data.

**返回**

- `best_loss` (`float`): Best training loss achieved.
- `best_collision_rate` (`float`): Best collision rate achieved during evaluation.

##### `Trainer.export_onnx`

```python
export_onnx(self, output_path, batch_size = 2, opset_version = 14, dynamic_batch = True, device = None, verbose = False, onnx_export_kwargs = None)
```

Export the trained RQVAE model to ONNX format, including reconstructed output and codebook indices.

**参数**

- `output_path` (`str`): Path to save the ONNX model.
- `batch_size` (`int, optional`): Batch size for the dummy input used in export.
- `opset_version` (`int, optional`): ONNX opset version.
- `dynamic_batch` (`bool, optional`): Whether to enable dynamic batch size.
- `device` (`torch.device or str, optional`): Device to run the export (cpu or cuda). Default: model device.
- `verbose` (`bool, optional`): Whether to print ONNX export debug info.
- `onnx_export_kwargs` (`dict, optional`): Additional kwargs for torch.onnx.export.

**返回**

- `bool`: True if export succeeded, False otherwise.

**示例**

```python
>>> model = RQVAEModel(in_dim=768, num_emb_list=[64,64], e_dim=64)
>>> model.train()  # assume model has been trained
>>> output_path = "rqevae.onnx"
>>> success = model.export_onnx(output_path, batch_size=4, opset_version=14)
>>> print(success)
```
True

```python
>>> # Export on specific device
>>> success = model.export_onnx("rqevae_cpu.onnx", batch_size=4, device="cpu")
>>> print(success)
```
True

### `seq_trainer`

模块: `torch_rechub.trainers.seq_trainer`

#### `SeqTrainer`

Sequence Generation Model Trainer.

Used for training HSTU, HLLM, and RQVAE models.
Supports CrossEntropyLoss, NCE Loss, and RQVAE-specific losses.

**参数**

- `model` (`nn.Module`): Model to be trained.
- `optimizer_fn` (`torch.optim`): Optimizer constructor, default torch.optim.Adam.
- `optimizer_params` (`dict`): Optimizer parameters.
- `scheduler_fn` (`torch.optim.lr_scheduler`): Torch scheduler class.
- `scheduler_params` (`dict`): Scheduler parameters.
- `n_epoch` (`int`): Number of training epochs, default 10.
- `earlystop_patience` (`int`): Early stopping patience, default 10.
- `device` (`str`): Device 'cpu' or 'cuda', default 'cpu'.
- `gpus` (`list`): List of GPU ids for parallel training, default [].
- `model_path` (`str`): Path to save model checkpoints, default './'.
- `loss_type` (`str`): Loss function type ('cross_entropy', 'nce', 'mse', 'l1').
- `loss_params` (`dict`): Parameters for loss function.
- `model_logger` (`object`): Logger instance.
- `eval_step` (`int`): Evaluation interval in epochs (for RQVAE), default 50.

##### `SeqTrainer.fit`

```python
fit(self, train_dataloader, val_dataloader = None)
```

Train the model.

**参数**

- `train_dataloader` (`DataLoader`): Training data loader.
- `val_dataloader` (`DataLoader`): Validation data loader.

**返回**

- `dict or tuple: Training history or (best_loss, best_collision_rate) for RQVAE.`

##### `SeqTrainer.train_rqvae_one_epoch`

```python
train_rqvae_one_epoch(self, data_loader)
```

Train one epoch for RQVAE model.

##### `SeqTrainer.evaluate_rqvae`

```python
evaluate_rqvae(self, data_loader)
```

Evaluate RQVAE model (calculate collision rate).

##### `SeqTrainer.train_one_epoch`

```python
train_one_epoch(self, data_loader, log_interval = 10)
```

Train the model for a single epoch.

**参数**

- `data_loader` (`DataLoader`): Training data loader.
- `log_interval` (`int`): Interval (in steps) for logging average loss.

**返回**

- `float` (`Average training loss for this epoch.`)

##### `SeqTrainer.evaluate`

```python
evaluate(self, data_loader)
```

Evaluate the model on a validation/test data loader.

**参数**

- `data_loader` (`DataLoader`): Validation or test data loader.

**返回**

- `tuple` (```(avg_loss, top1_accuracy)``.`)

##### `SeqTrainer.export_onnx`

```python
export_onnx(self, output_path, batch_size = 2, seq_length = 50, vocab_size = None, opset_version = 14, dynamic_batch = True, device = None, verbose = False, onnx_export_kwargs = None)
```

Export the trained sequence generation model to ONNX format.

This method exports sequence generation models (e.g., HSTU) to ONNX format.
Unlike other trainers, sequence models use positional arguments (seq_tokens, seq_time_diffs)
instead of dict input, making ONNX export more straightforward.

**参数**

- `output_path` (`str`): Path to save the ONNX model file.
- `batch_size` (`int`): Batch size for dummy input (default: 2).
- `seq_length` (`int`): Sequence length for dummy input (default: 50).
- `vocab_size` (`int, optional`): Vocabulary size for generating dummy tokens.
  If None, will try to get from model.vocab_size.
- `opset_version` (`int`): ONNX opset version (default: 14).
- `dynamic_batch` (`bool`): Enable dynamic batch size (default: True).
- `device` (`str, optional`): Device for export ('cpu', 'cuda', etc.).
  If None, defaults to 'cpu' for maximum compatibility.
- `verbose` (`bool`): Print export details (default: False).
- `onnx_export_kwargs` (`dict, optional`): Extra kwargs forwarded to ``torch.onnx.export``.

**返回**

- `bool` (`True if export succeeded, False otherwise.`)

**示例**

```python
>>> trainer = SeqTrainer(hstu_model, ...)
>>> trainer.fit(train_dl, val_dl)
>>> trainer.export_onnx("hstu.onnx", vocab_size=10000)

>>> # Export on specific device
>>> trainer.export_onnx("hstu.onnx", vocab_size=10000, device="cpu")
```

##### `SeqTrainer.visualization`

```python
visualization(self, seq_length = 50, vocab_size = None, batch_size = 2, depth = 3, show_shapes = True, expand_nested = True, save_path = None, graph_name = 'model', device = None, dpi = 300, **kwargs)
```

Visualize the model's computation graph.

This method generates a visual representation of the sequence model
architecture, showing layer connections, tensor shapes, and nested
module structures.

**参数**

- `seq_length` (`int, default=50`): Sequence length for dummy input.
- `vocab_size` (`int, optional`): Vocabulary size for generating dummy tokens.
  If None, will try to get from model.vocab_size or model.item_num.
- `batch_size` (`int, default=2`): Batch size for dummy input.
- `depth` (`int, default=3`): Visualization depth, higher values show more detail.
  Set to -1 to show all layers.
- `show_shapes` (`bool, default=True`): Whether to display tensor shapes.
- `expand_nested` (`bool, default=True`): Whether to expand nested modules.
- `save_path` (`str, optional`): Path to save the graph image (.pdf, .svg, .png).
  If None, displays in Jupyter or opens system viewer.
- `graph_name` (`str, default="model"`): Name for the graph.
- `device` (`str, optional`): Device for model execution. If None, defaults to 'cpu'.
- `dpi` (`int, default=300`): Resolution in dots per inch for output image.
  Higher values produce sharper images suitable for papers.
- `**kwargs` (`dict`): Additional arguments passed to torchview.draw_graph().

**返回**

- `ComputationGraph`: A torchview ComputationGraph object.

**异常**

- `ImportError`: If torchview or graphviz is not installed.
- `ValueError`: If vocab_size is not provided and cannot be inferred from model.

**说明**

Default Display Behavior:
    When `save_path` is None (default):
    - In Jupyter/IPython: automatically displays the graph inline
    - In Python script: opens the graph with system default viewer

**示例**

```python
>>> trainer = SeqTrainer(hstu_model, ...)
>>> trainer.fit(train_dl, val_dl)
>>>
>>> # Auto-display in Jupyter (no save_path needed)
>>> trainer.visualization(depth=4, vocab_size=10000)
>>>
>>> # Save to high-DPI PNG for papers
>>> trainer.visualization(save_path="model.png", dpi=300)
```

## `utils/`

### `data`

模块: `torch_rechub.utils.data`

#### `get_auto_embedding_dim`

```python
get_auto_embedding_dim(num_classes)
```

Calculate embedding dim by category size.

Uses ``emb_dim = floor(6 * num_classes**0.25)`` from DCN (ADKDD'17).

**参数**

- `num_classes` (`int`): Number of categorical classes.

**返回**

- `int`: Recommended embedding dimension.

#### `get_loss_func`

```python
get_loss_func(task_type = 'classification')
```

Return default loss by task type.

#### `get_metric_func`

```python
get_metric_func(task_type = 'classification')
```

Return default metric by task type.

#### `generate_seq_feature`

```python
generate_seq_feature(data, user_col, item_col, time_col, item_attribute_cols = [], min_item = 0, shuffle = True, max_len = 50)
```

Generate sequence features and negatives for ranking.

**参数**

- `data` (`pd.DataFrame`): Raw interaction data.
- `user_col` (`str`): User id column name.
- `item_col` (`str`): Item id column name.
- `time_col` (`str`): Timestamp column name.
- `item_attribute_cols` (`list[str], optional`): Additional item attribute columns to include in sequences.
- `min_item` (`int, default=0`): Minimum items per user; users below are dropped.
- `shuffle` (`bool, default=True`): Shuffle train/val/test.
- `max_len` (`int, default=50`): Max history length.

**返回**

- `tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Train, validation, and test data with sequence features.

#### `df_to_dict`

```python
df_to_dict(data)
```

Convert DataFrame to dict inputs accepted by models.

**参数**

- `data` (`pd.DataFrame`): Input dataframe.

**返回**

- `dict`: Mapping of column name to numpy array.

#### `neg_sample`

```python
neg_sample(click_hist, item_size)
```

未提供文档说明。

#### `pad_sequences`

```python
pad_sequences(sequences, maxlen = None, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.0)
```

Pad list-of-lists sequences to equal length.

Equivalent to ``tf.keras.preprocessing.sequence.pad_sequences``.

**参数**

- `sequences` (`Sequence[Sequence]`): Input sequences.
- `maxlen` (`int, optional`): Maximum length; computed if None.
- `dtype` (`str, default='int32'`)
- `padding` (`{'pre', 'post'}, default='pre'`): Padding direction.
- `truncating` (`{'pre', 'post'}, default='pre'`): Truncation direction.
- `value` (`float, default=0.0`): Padding value.

**返回**

- `np.ndarray`: Padded array of shape (n_samples, maxlen).

#### `array_replace_with_dict`

```python
array_replace_with_dict(array, dic)
```

Replace values in numpy array using a mapping dict.

**参数**

- `array` (`np.ndarray`): Input array.
- `dic` (`dict`): Mapping from old to new values.

**返回**

- `np.ndarray`: Array with values replaced.

#### `create_seq_features`

```python
create_seq_features(data, seq_feature_col = ['item_id', 'cate_id'], max_len = 50, drop_short = 3, shuffle = True)
```

Build user history sequences by time.

**参数**

- `data` (`pd.DataFrame`): Must contain ``user_id, item_id, cate_id, time``.
- `seq_feature_col` (`list, default ['item_id', 'cate_id']`): Columns to generate sequence features.
- `max_len` (`int, default=50`): Max history length.
- `drop_short` (`int, default=3`): Drop users with sequence length < drop_short.
- `shuffle` (`bool, default=True`): Shuffle outputs.

**返回**

- `tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Train/val/test splits with sequence features.

#### `TorchDataset`

未提供文档说明。

#### `PredictDataset`

未提供文档说明。

#### `MatchDataGenerator`

未提供文档说明。

##### `MatchDataGenerator.generate_dataloader`

```python
generate_dataloader(self, x_test_user, x_all_item, batch_size, num_workers = 8)
```

未提供文档说明。

#### `DataGenerator`

未提供文档说明。

##### `DataGenerator.generate_dataloader`

```python
generate_dataloader(self, x_val = None, y_val = None, x_test = None, y_test = None, split_ratio = None, batch_size = 16, num_workers = 0)
```

未提供文档说明。

#### `SeqDataset`

Sequence dataset for HSTU-style next-item prediction.

**参数**

- `seq_tokens` (`np.ndarray`): Token ids, shape ``(num_samples, seq_len)``.
- `seq_positions` (`np.ndarray`): Position indices, shape ``(num_samples, seq_len)``.
- `targets` (`np.ndarray`): Target token ids, shape ``(num_samples,)``.
- `seq_time_diffs` (`np.ndarray`): Time-difference features, shape ``(num_samples, seq_len)``.

**张量形状**

Output tuple: ``(seq_tokens, seq_positions, seq_time_diffs, target)``

**示例**

```python
>>> seq_tokens = np.random.randint(0, 1000, (100, 256))
>>> seq_positions = np.arange(256)[np.newaxis, :].repeat(100, axis=0)
>>> seq_time_diffs = np.random.randint(0, 86400, (100, 256))
>>> targets = np.random.randint(0, 1000, (100,))
>>> dataset = SeqDataset(seq_tokens, seq_positions, targets, seq_time_diffs)
>>> len(dataset)
```
100

#### `SequenceDataGenerator`

Sequence data generator for HSTU-style models.

Wraps :class:`SeqDataset` and builds train/val/test loaders.

**参数**

- `seq_tokens` (`np.ndarray`): Token ids, shape ``(num_samples, seq_len)``.
- `seq_positions` (`np.ndarray`): Position indices, shape ``(num_samples, seq_len)``.
- `targets` (`np.ndarray`): Target token ids, shape ``(num_samples,)``.
- `seq_time_diffs` (`np.ndarray`): Time-difference features, shape ``(num_samples, seq_len)``.

**示例**

```python
>>> gen = SequenceDataGenerator(seq_tokens, seq_positions, targets, seq_time_diffs)
>>> train_loader, val_loader, test_loader = gen.generate_dataloader(batch_size=32)
```

##### `SequenceDataGenerator.generate_dataloader`

```python
generate_dataloader(self, batch_size = 32, num_workers = 0, split_ratio = None, shuffle = True)
```

Generate dataloader(s) from the dataset.

**参数**

- `batch_size` (`int, default=32`): Batch size for DataLoader.
- `num_workers` (`int, default=0`): Number of workers for DataLoader.
- `split_ratio` (`tuple or None, default=None`): If None, returns a single DataLoader without splitting the data.
  If tuple (e.g., (0.7, 0.1, 0.2)), splits dataset and returns
  (train_loader, val_loader, test_loader).
- `shuffle` (`bool, default=True`): Whether to shuffle data. Only applies when split_ratio is None.
  When split_ratio is provided, train data is always shuffled.

**返回**

- `tuple`: If split_ratio is None: returns (dataloader,)
  If split_ratio is provided: returns (train_loader, val_loader, test_loader)

**示例**

**Case 1: Data already split, just create loader**
```python
>>> train_gen = SequenceDataGenerator(train_data['seq_tokens'], ...)
>>> train_loader = train_gen.generate_dataloader(batch_size=32)[0]

```
**Case 2: Auto-split data into train/val/test**
```python
>>> all_gen = SequenceDataGenerator(all_data['seq_tokens'], ...)
>>> train_loader, val_loader, test_loader = all_gen.generate_dataloader(
...     batch_size=32, split_ratio=(0.7, 0.1, 0.2))
```

#### `EmbDataset`

Embedding dataset for loading precomputed feature vectors.

Loads embeddings stored in ``.npy`` or ``.pt`` format and exposes
them as a PyTorch ``Dataset`` for downstream training or inference.

**参数**

- `data_path` (`str`): Path to the embedding file. Supported formats are ``.npy``
  (NumPy array) and ``.pt`` (PyTorch tensor saved via ``torch.save``).
- `device` (`str, default='cpu'`): Device used when loading ``.pt`` tensors.

**张量形状**

- **Input**
  - `embeddings` (```(num_samples, emb_dim)```)
- **Output**
  - `tensor_emb` (```(emb_dim,)```)

**示例**

```python
>>> dataset = EmbDataset("embeddings.npy")
>>> len(dataset)
```
10000
```python
>>> emb = dataset[0]
>>> emb.shape
```
torch.Size([768])

#### `TigerSeqDataset`

未提供文档说明。

##### `TigerSeqDataset.get_new_tokens`

```python
get_new_tokens(self)
```

未提供文档说明。

##### `TigerSeqDataset.get_all_items`

```python
get_all_items(self, as_list = False)
```

未提供文档说明。

##### `TigerSeqDataset.get_prefix_allowed_tokens_fn`

```python
get_prefix_allowed_tokens_fn(self, tokenizer)
```

未提供文档说明。

##### `TigerSeqDataset.get_collate_fn`

```python
get_collate_fn(self, tokenizer)
```

未提供文档说明。

#### `Trie`

未提供文档说明。

##### `Trie.def_prefix_allowed_tokens_fn`

```python
def_prefix_allowed_tokens_fn(self, candidate_trie)
```

未提供文档说明。

##### `Trie.append`

```python
append(self, trie, bos_token_id)
```

未提供文档说明。

##### `Trie.add`

```python
add(self, sequence)
```

未提供文档说明。

##### `Trie.get`

```python
get(self, prefix_sequence)
```

未提供文档说明。

##### `Trie.load_from_dict`

```python
load_from_dict(trie_dict)
```

未提供文档说明。

### `hstu_utils`

模块: `torch_rechub.utils.hstu_utils`

#### `RelPosBias`

Relative position bias for attention.

**参数**

- `n_heads` (`int`): Number of attention heads.
- `max_seq_len` (`int`): Maximum supported sequence length.
- `num_buckets` (`int, default=32`): Number of relative position buckets.

**张量形状**

Output: ``(1, n_heads, seq_len, seq_len)``

**示例**

```python
>>> rel_pos_bias = RelPosBias(n_heads=8, max_seq_len=256)
>>> bias = rel_pos_bias(256)
>>> bias.shape
```
torch.Size([1, 8, 256, 256])

##### `RelPosBias.forward`

```python
forward(self, seq_len)
```

Compute relative position bias for a given sequence length.

**参数**

- `seq_len` (`int`): Sequence length ``L``.

**返回**

- `Tensor` (`Relative position bias of shape ``(1, n_heads, L, L)``.`)

#### `VocabMask`

Vocabulary mask to block invalid items at inference.

**参数**

- `vocab_size` (`int`): Vocabulary size.
- `invalid_items` (`list, optional`): IDs to mask out.

**示例**

```python
>>> mask = VocabMask(vocab_size=1000, invalid_items=[0, 1, 2])
>>> logits = torch.randn(32, 1000)
>>> masked_logits = mask.apply_mask(logits)
```

##### `VocabMask.apply_mask`

```python
apply_mask(self, logits)
```

Apply mask to logits.

**参数**

- `logits` (`Tensor`): Model logits, shape ``(..., vocab_size)``.

**返回**

- `Tensor`: Masked logits.

#### `VocabMapper`

Identity mapper between ``item_id`` and ``token_id``.

Useful for sequence generation where items are treated as tokens.

**参数**

- `vocab_size` (`int`): Vocabulary size.
- `pad_id` (`int, default=0`): PAD token id.
- `unk_id` (`int, default=1`): Unknown token id.

**示例**

```python
>>> mapper = VocabMapper(vocab_size=1000)
>>> item_ids = np.array([10, 20, 30])
>>> token_ids = mapper.encode(item_ids)
>>> decoded_ids = mapper.decode(token_ids)
```

##### `VocabMapper.encode`

```python
encode(self, item_ids)
```

Convert item_ids to token_ids.

**参数**

- `item_ids` (`np.ndarray`): Item ids.

**返回**

- `np.ndarray`: Token ids.

##### `VocabMapper.decode`

```python
decode(self, token_ids)
```

Convert token_ids back to item_ids.

**参数**

- `token_ids` (`np.ndarray`): Token ids.

**返回**

- `np.ndarray`: Item ids.

### `match`

模块: `torch_rechub.utils.match`

#### `gen_model_input`

```python
gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len, padding = 'pre', truncating = 'pre')
```

Merge user_profile and item_profile to df, pad and truncate history sequence feature.

**参数**

- `df` (`pd.DataFrame`): data with history sequence feature
- `user_profile` (`pd.DataFrame`): user data
- `user_col` (`str`): user column name
- `item_profile` (`pd.DataFrame`): item data
- `item_col` (`str`): item column name
- `seq_max_len` (`int`): sequence length of every data
- `padding` (`str, optional`): padding style, {'pre', 'post'}. Defaults to 'pre'.
- `truncating` (`str, optional`): truncate style, {'pre', 'post'}. Defaults to 'pre'.

**返回**

- `dict` (`The converted dict, which can be used directly into the input network`)

#### `negative_sample`

```python
negative_sample(items_cnt_order, ratio, method_id = 0)
```

Negative Sample method for matching model.

Reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py
Updated with more methods and redesigned this function.

**参数**

- `items_cnt_order` (`dict`): the item count dict, the keys(item) sorted by value(count) in reverse order.
- `ratio` (`int`): negative sample ratio, >= 1
- `method_id` (`int, optional`)
- `0` (`"random sampling",`)
- `1` (`"popularity sampling method used in word2vec",`)
- `2` (`"popularity sampling method by `log(count+1)+1e-6`",`)
- `3` (`"tencent RALM sampling"}`.`)

    `{
        Defaults to 0.

**返回**

- `list` (`sampled negative item list`)

#### `inbatch_negative_sampling`

```python
inbatch_negative_sampling(scores, neg_ratio = None, hard_negative = False, generator = None)
```

Generate in-batch negative indices from a similarity matrix.

This mirrors the offline ``negative_sample`` API by only returning sampled
indices; score gathering is handled separately to keep responsibilities clear.

**参数**

- `scores` (`torch.Tensor`): similarity matrix with shape (batch_size, batch_size).
- `neg_ratio` (`int, optional`): number of negatives for each positive sample.
  Defaults to batch_size-1 when omitted or out of range.
- `hard_negative` (`bool, optional`): whether to pick top-k highest scores as negatives
  instead of uniform random sampling. Defaults to False.
- `generator` (`torch.Generator, optional`): generator to control randomness for tests/reproducibility.

**返回**

- `torch.Tensor` (`sampled negative indices with shape (batch_size, neg_ratio).`)

#### `gather_inbatch_logits`

```python
gather_inbatch_logits(scores, neg_indices)
```

scores: (B, B)
    scores[i][j] = user_i ⋅ item_j
neg_indices: (B, K)
    neg_indices[i] = the K negative items for user_i

#### `generate_seq_feature_match`

```python
generate_seq_feature_match(data, user_col, item_col, time_col, item_attribute_cols = None, sample_method = 0, mode = 0, neg_ratio = 0, min_item = 0)
```

Generate sequence feature and negative sample for match.

**参数**

- `data` (`pd.DataFrame`): the raw data.
- `user_col` (`str`): the col name of user_id
- `item_col` (`str`): the col name of item_id
- `time_col` (`str`): the col name of timestamp
- `item_attribute_cols` (`list[str], optional`): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
- `sample_method` (`int, optional`): the negative sample method `{
  0: "random sampling",
  1: "popularity sampling method used in word2vec",
  2: "popularity sampling method by `log(count+1)+1e-6`",
  3: "tencent RALM sampling"}`.
  Defaults to 0.
- `mode` (`int, optional`): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
- `neg_ratio` (`int, optional`): negative sample ratio, >= 1. Defaults to 0.
- `min_item` (`int, optional`): the min item each user must have. Defaults to 0.

**返回**

- `pd.DataFrame` (`split train and test data with sequence features.`)

#### `Annoy`

A vector matching engine using Annoy library

##### `Annoy.fit`

```python
fit(self, X)
```

Build the Annoy index from input vectors.

**参数**

- `X` (`np.ndarray`): input vectors with shape (n_samples, n_features)

##### `Annoy.set_query_arguments`

```python
set_query_arguments(self, search_k)
```

Set query parameters for searching.

**参数**

- `search_k` (`int`): number of nodes to inspect during searching

##### `Annoy.query`

```python
query(self, v, n)
```

Find the n nearest neighbors to vector v.

**参数**

- `v` (`np.ndarray`): query vector
- `n` (`int`): number of nearest neighbors to return

**返回**

- `tuple` (`(indices, distances) - lists of nearest neighbor indices and their distances`)

#### `Milvus`

A vector matching engine using Milvus database

##### `Milvus.fit`

```python
fit(self, X)
```

Insert vectors into Milvus collection and build index.

**参数**

- `X` (`np.ndarray or torch.Tensor`): input vectors with shape (n_samples, n_features)

##### `Milvus.process_result`

```python
process_result(results)
```

Process Milvus search results into standard format.

**参数**

- `results` (`raw search results from Milvus`)

**返回**

- `tuple` (`(indices_list, distances_list) - processed results`)

##### `Milvus.query`

```python
query(self, v, n)
```

Query Milvus for the n nearest neighbors to vector v.

**参数**

- `v` (`np.ndarray or torch.Tensor`): query vector
- `n` (`int`): number of nearest neighbors to return

**返回**

- `tuple` (`(indices, distances) - lists of nearest neighbor indices and their distances`)

#### `Faiss`

A vector matching engine using Faiss library

##### `Faiss.fit`

```python
fit(self, X)
```

Train and build the index from input vectors.

**参数**

- `X` (`np.ndarray`): input vectors with shape (n_samples, dim)

##### `Faiss.query`

```python
query(self, v, n)
```

Query the nearest neighbors for given vector.

**参数**

- `v` (`np.ndarray or torch.Tensor`): query vector
- `n` (`int`): number of nearest neighbors to return

**返回**

- `tuple` (`(indices, distances) - lists of nearest neighbor indices and distances`)

##### `Faiss.set_query_arguments`

```python
set_query_arguments(self, nprobe = None, efSearch = None)
```

Set query parameters for search.

**参数**

- `nprobe` (`int`): number of clusters to search for IVF index
- `efSearch` (`int`): search parameter for HNSW index

##### `Faiss.save_index`

```python
save_index(self, filepath)
```

Save index to file for later use.

##### `Faiss.load_index`

```python
load_index(self, filepath)
```

Load index from file.

### `model_utils`

模块: `torch_rechub.utils.model_utils`

#### `extract_feature_info`

```python
extract_feature_info(model: nn.Module) -> Dict[str, Any]
```

Extract feature information from a torch-rechub model via reflection.

**参数**

- `model` (`nn.Module`): Model to inspect.

**返回**

- `dict`: {
  'features': list of unique Feature objects,
  'input_names': ordered feature names,
  'input_types': map name -> feature type,
  'user_features': user-side features (dual-tower),
  'item_features': item-side features (dual-tower),
  }

**示例**

```python
>>> from torch_rechub.models.ranking import DeepFM
>>> model = DeepFM(deep_features, fm_features, mlp_params)
>>> info = extract_feature_info(model)
>>> info['input_names']  # ['user_id', 'item_id', ...]
```

#### `generate_dummy_input`

```python
generate_dummy_input(features: List[Any], batch_size: int = 2, seq_length: int = 10, device: str = 'cpu') -> Tuple[torch.Tensor, Ellipsis]
```

Generate dummy input tensors based on feature definitions.

**参数**

- `features` (`list`): List of Feature objects (SparseFeature, DenseFeature, SequenceFeature).
- `batch_size` (`int, default=2`): Batch size for dummy input.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature.
- `device` (`str, default='cpu'`): Device to create tensors on.

**返回**

- `tuple of Tensor`: Tuple of tensors in the order of input features.

**示例**

```python
>>> features = [SparseFeature("user_id", 1000), SequenceFeature("hist", 500)]
>>> dummy = generate_dummy_input(features, batch_size=4)
>>> # Returns (user_id_tensor[4], hist_tensor[4, 10])
```

#### `generate_dummy_input_dict`

```python
generate_dummy_input_dict(features: List[Any], batch_size: int = 2, seq_length: int = 10, device: str = 'cpu') -> Dict[str, torch.Tensor]
```

Generate dummy input dict based on feature definitions.

Similar to generate_dummy_input but returns a dict mapping feature names
to tensors. This is the expected input format for torch-rechub models.

**参数**

- `features` (`list`): List of Feature objects (SparseFeature, DenseFeature, SequenceFeature).
- `batch_size` (`int, default=2`): Batch size for dummy input.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature.
- `device` (`str, default='cpu'`): Device to create tensors on.

**返回**

- `dict`: Dict mapping feature names to tensors.

**示例**

```python
>>> features = [SparseFeature("user_id", 1000)]
>>> dummy = generate_dummy_input_dict(features, batch_size=4)
>>> # Returns {"user_id": tensor[4]}
```

#### `generate_dynamic_axes`

```python
generate_dynamic_axes(input_names: List[str], output_names: Optional[List[str]] = None, batch_dim: int = 0, include_seq_dim: bool = True, seq_features: Optional[List[str]] = None) -> Dict[str, Dict[int, str]]
```

Generate dynamic axes configuration for ONNX export.

**参数**

- `input_names` (`list of str`): List of input tensor names.
- `output_names` (`list of str, optional`): List of output tensor names. Default is ["output"].
- `batch_dim` (`int, default=0`): Dimension index for batch size.
- `include_seq_dim` (`bool, default=True`): Whether to include sequence dimension as dynamic.
- `seq_features` (`list of str, optional`): List of feature names that are sequences.

**返回**

- `dict`: Dynamic axes dict for torch.onnx.export.

**示例**

```python
>>> axes = generate_dynamic_axes(["user_id", "item_id"], seq_features=["hist"])
>>> # Returns {"user_id": {0: "batch_size"}, "item_id": {0: "batch_size"}, ...}
```

### `mtl`

模块: `torch_rechub.utils.mtl`

#### `shared_task_layers`

```python
shared_task_layers(model)
```

get shared layers and task layers in multi-task model
Authors: Qida Dong, dongjidan@126.com

**参数**

- `model` (`torch.nn.Module`): only support `[MMOE, SharedBottom, PLE, AITM]`

**返回**

- `list[torch.nn.parameter]: parameters split to shared list and task list.`

#### `gradnorm`

```python
gradnorm(loss_list, loss_weight, share_layer, initial_task_loss, alpha)
```

未提供文档说明。

#### `MetaBalance`

MetaBalance Optimizer
This method is used to scale the gradient and balance the gradient of each task.
Authors: Qida Dong, dongjidan@126.com

**参数**

- `parameters` (`list`): the parameters of model
- `relax_factor` (`float, optional`): the relax factor of gradient scaling (default: 0.7)
- `beta` (`float, optional`): the coefficient of moving average (default: 0.9)

##### `MetaBalance.step`

```python
step(self, losses)
```

未提供文档说明。

### `onnx_export`

模块: `torch_rechub.utils.onnx_export`

#### `ONNXWrapper`

Wrap a dict-input model to accept positional args for ONNX.

ONNX disallows dict inputs; this wrapper maps positional args back to dict
before calling the original model.

**参数**

- `model` (`nn.Module`): Original dict-input model.
- `input_names` (`list[str]`): Ordered feature names matching positional inputs.
- `mode` (`{'user', 'item'}, optional`): For dual-tower models, set tower mode.

**示例**

```python
>>> wrapper = ONNXWrapper(dssm_model, ["user_id", "movie_id", "hist_movie_id"])
>>> wrapper(user_id_tensor, movie_id_tensor, hist_tensor)
```

##### `ONNXWrapper.forward`

```python
forward(self, *args) -> torch.Tensor
```

Convert positional args to dict and call original model.

##### `ONNXWrapper.restore_mode`

```python
restore_mode(self)
```

Restore the original mode of the model.

#### `ONNXExporter`

Main class for exporting Torch-RecHub models to ONNX format.

This exporter handles the complexity of converting dict-input models to ONNX
by automatically extracting feature information and wrapping the model.

**参数**

- `model` (`The PyTorch recommendation model to export.`)
- `device` (`Device for export operations (default: 'cpu').`)

**示例**

```python
>>> exporter = ONNXExporter(deepfm_model)
>>> exporter.export("model.onnx")

>>> # For dual-tower models
>>> exporter = ONNXExporter(dssm_model)
>>> exporter.export("user_tower.onnx", mode="user")
>>> exporter.export("item_tower.onnx", mode="item")
```

##### `ONNXExporter.export`

```python
export(self, output_path: str, mode: Optional[str] = None, dummy_input: Optional[Dict[str, torch.Tensor]] = None, batch_size: int = 2, seq_length: int = 10, opset_version: int = 14, dynamic_batch: bool = True, verbose: bool = False, onnx_export_kwargs: Optional[Dict[str, Any]] = None) -> bool
```

Export model to ONNX format.

**参数**

- `output_path` (`str`): Destination path.
- `mode` (`{'user', 'item'}, optional`): For dual-tower, export specific tower; None exports full model.
- `dummy_input` (`dict[str, Tensor], optional`): Example inputs; auto-generated if None.
- `batch_size` (`int, default=2`): Batch size for dummy input generation.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature.
- `opset_version` (`int, default=14`): ONNX opset.
- `dynamic_batch` (`bool, default=True`): Enable dynamic batch axes.
- `verbose` (`bool, default=False`): Print export details.
- `onnx_export_kwargs` (`dict, optional`): Extra keyword args forwarded to ``torch.onnx.export`` (e.g. ``operator_export_type``,
  ``keep_initializers_as_inputs``, ``do_constant_folding``).

**说明**

      - If you pass keys that overlap with the explicit parameters above
        (like ``opset_version`` / ``dynamic_axes`` / ``input_names``), this function
        will raise a ``ValueError`` to avoid ambiguous behavior.
      - By default, this method tries the dynamo exporter first and
        falls back to the legacy exporter automatically. To force one
        path, pass ``onnx_export_kwargs={"dynamo": True}`` or
        ``onnx_export_kwargs={"dynamo": False}``.
      - Some kwargs (like ``dynamo``) are only available in newer PyTorch; unsupported
        keys will be ignored for compatibility.

**返回**

- `bool`: True if export succeeds.

**异常**

- `RuntimeError`: If ONNX export fails.

##### `ONNXExporter.get_input_info`

```python
get_input_info(self, mode: Optional[str] = None) -> Dict[str, Any]
```

Get information about model inputs.

**参数**

- `mode` (`For dual-tower models, "user" or "item".`)

**返回**

- `Dict with input names, types, and shapes.`

### `quantization`

模块: `torch_rechub.utils.quantization`

#### `quantize_model`

```python
quantize_model(input_path: str, output_path: str, mode: str = 'int8', *, per_channel: bool = False, reduce_range: bool = False, weight_type: str = 'qint8', optimize_model: bool = False, op_types_to_quantize: Optional[list[str]] = None, nodes_to_quantize: Optional[list[str]] = None, nodes_to_exclude: Optional[list[str]] = None, extra_options: Optional[Dict[str, Any]] = None, keep_io_types: bool = True) -> str
```

Quantize an ONNX model.

**参数**

- `input_path` (`str`): Input ONNX model path (FP32).
- `output_path` (`str`): Output ONNX model path.
- `mode` (`str, default="int8"`): Quantization mode:
  - "int8" / "dynamic_int8": ONNX Runtime dynamic quantization (weights INT8).
  - "fp16": convert float tensors to float16.
- `per_channel` (`bool, default=False`): Enable per-channel quantization for weights (INT8).
- `reduce_range` (`bool, default=False`): Use reduced quantization range (INT8), sometimes helpful on certain CPUs.
- `weight_type` (`{"qint8", "quint8"}, default="qint8"`): Weight quant type for dynamic quantization.
- `optimize_model` (`bool, default=False`): Run ORT graph optimization before quantization.
- `keep_io_types` (`bool, default=True`): For FP16 conversion, keep model input/output types as float32 for compatibility.


op_types_to_quantize / nodes_to_quantize / nodes_to_exclude / extra_options
    Advanced options forwarded to ``onnxruntime.quantization.quantize_dynamic``.

**返回**

- `str`: The output_path.

### `visualization`

模块: `torch_rechub.utils.visualization`

#### `display_graph`

```python
display_graph(graph: Any, format: str = 'png') -> Any
```

Display a torchview ComputationGraph in Jupyter.

**参数**

- `graph` (`ComputationGraph`): Returned by :func:`visualize_model`.
- `format` (`str, default='png'`): Output format; 'png' recommended for VSCode.

**返回**

- `graphviz.Digraph or None`: Displayed graph object, or None if display fails.

#### `visualize_model`

```python
visualize_model(model: nn.Module, input_data: Optional[Dict[str, torch.Tensor]] = None, batch_size: int = 2, seq_length: int = 10, depth: int = 3, show_shapes: bool = True, expand_nested: bool = True, save_path: Optional[str] = None, graph_name: str = 'model', device: str = 'cpu', dpi: int = 300, **kwargs) -> Any
```

Visualize a Torch-RecHub model's computation graph.

This function generates a visual representation of the model architecture,
showing layer connections, tensor shapes, and nested module structures.
It automatically extracts feature information from the model to generate
appropriate dummy inputs.

**参数**

- `model` (`nn.Module`): PyTorch model to visualize. Should be a Torch-RecHub model
  with feature attributes (e.g., DeepFM, DSSM, MMOE).
- `input_data` (`dict, optional`): Dict of example inputs {feature_name: tensor}.
  If None, inputs are auto-generated based on model features.
- `batch_size` (`int, default=2`): Batch size for auto-generated inputs.
- `seq_length` (`int, default=10`): Sequence length for SequenceFeature inputs.
- `depth` (`int, default=3`): Visualization depth - higher values show more detail.
  Set to -1 to show all layers.
- `show_shapes` (`bool, default=True`): Whether to display tensor shapes on edges.
- `expand_nested` (`bool, default=True`): Whether to expand nested nn.Module with dashed borders.
- `save_path` (`str, optional`): Path to save the graph image. Supports .pdf, .svg, .png formats.
  If None, displays in Jupyter or opens system viewer.
- `graph_name` (`str, default="model"`): Name for the computation graph.
- `device` (`str, default="cpu"`): Device for model execution during tracing.
- `dpi` (`int, default=300`): Resolution in dots per inch for output image.
  Higher values produce sharper images suitable for papers.
- `**kwargs` (`dict`): Additional arguments passed to torchview.draw_graph().

**返回**

- `ComputationGraph`: A torchview ComputationGraph object.
  - Use `.visual_graph` property to get the graphviz.Digraph
  - Use `.resize_graph(scale=1.5)` to adjust graph size

**异常**

- `ImportError`: If torchview or graphviz is not installed.
- `ValueError`: If model has no recognizable feature attributes.

**说明**

Default Display Behavior:
    When `save_path` is None (default):
    - In Jupyter/IPython: automatically displays the graph inline
    - In Python script: opens the graph with system default viewer

Requires graphviz system package: apt/brew/choco install graphviz.
For Jupyter display issues, try: graphviz.set_jupyter_format('png').

**示例**

```python
>>> from torch_rechub.models.ranking import DeepFM
>>> from torch_rechub.utils.visualization import visualize_model
>>>
>>> # Auto-display in Jupyter or open in viewer
>>> visualize_model(model, depth=4)  # No save_path needed
>>>
>>> # Save to high-DPI PNG for paper
>>> visualize_model(model, save_path="model.png", dpi=300)
```
