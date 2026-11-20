---
title: 排序模型
description: Torch-RecHub 排序模型详细介绍
---

# 排序模型

排序模型给每一对用户和候选物品打分。例如，首页有十部候选电影，可以构造十行样本，用户特征相同，电影特征不同，按模型输出的十个分数排序。第一次使用可以从 [DeepFM 快速入门](/zh/guide/quick_start)开始，下面的各个示例侧重于怎样构造模型。

## 批内负采样：统一训练入口

手头有曝光和点击标签时，默认按原始标签训练即可。如果想用同一批用户点击过的物品构造负例，在 `CTRTrainer` 中指定哪些列属于物品。

```python
trainer = CTRTrainer(model, in_batch_neg=True,
                     item_features=["item_id", "cate_id"])
```

假设小王点了科幻片 A，小李点了喜剧片 B。训练器先找出这一批中标签为 1 的行，保留小王的用户特征、历史和上下文，把 B 的 `item_id` 和 `cate_id` 一起换进来，组成标签为 0 的新样本。如果只换电影 ID，小王的负样本就会变成“喜剧片 B，类别是科幻”。

`item_features` 可以写列名，也可以写已有的 Feature 对象。物品 ID 和随物品变化的属性都要列出。如果有“该用户近七天看过多少部同类影片”这样的交叉特征，换电影后需要重新计算，本入口不会代算。Criteo 的匿名列没有标注归属，使用前需要知道哪些列属于物品。

- 默认每个正样本随机配一个其他行的物品。同一行不会抽中自己，但不同的行可能含同一件物品。将 `in_batch_neg_ratio` 设为正整数可以有放回地抽取指定数量，设为 `None` 则使用所有其他正样本行。
- `sampler_seed` 控制抽样的随机性。批次中不足两条正样本时会跳过这一批。如果整轮都没有可以配对的样本，训练器会报错。
- Amazon DIN 和 DIEN 示例已定义好物品列，加 `--in_batch_neg` 即可切换。DIEN 仍需提供辅助损失用的负历史序列。其他单任务排序示例还要指定 `--item_features`。

合成的负例只表示“训练时把这两者配成负例”。评估点击率时，仍要用留出的真实曝光及其点击标签。有意生成纯正样本训练集时，可以使用 `generate_seq_feature(..., in_batch_neg=True)`，验证集和测试集保留原来的标签。

## 1. WideDeep

WideDeep 把一条曝光交给两个分支。Wide 用线性层给选定字段打分，Deep 把字段拼起来交给 MLP。两个分支的分数相加后，通过 sigmoid 得到点击分数。

### 论文引用

```
Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
```

代码里的 `wide_features` 和 `deep_features` 分别决定哪些字段进入哪个分支。

### 使用方法

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import WideDeep

# 定义特征
dense_features = [DenseFeature(name="age", embed_dim=1), DenseFeature(name="income", embed_dim=1)]
sparse_features = [SparseFeature(name="city", vocab_size=100, embed_dim=16), SparseFeature(name="gender", vocab_size=3, embed_dim=16)]

# 创建模型
model = WideDeep(
    wide_features=sparse_features,
    deep_features=sparse_features + dense_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| wide_features | list | Wide 部分使用的特征列表 | 必填 |
| deep_features | list | Deep 部分使用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数，包含 dims、dropout、activation 等 | 必填 |

## 2. DeepFM

DeepFM 让 FM 分支计算字段之间的两两交互，同时用 MLP 处理拼在一起的特征。例如同样是周末曝光，不同电影类型可以得到不同分数。

### 论文引用

```
Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." Proceedings of the 26th international joint conference on artificial intelligence. 2017.
```

FM 分支需要把各个字段叠在一起做点积，所以 `fm_features` 中稀疏特征的嵌入维度要相同。某个字段同时出现在两个分支时，共用它的嵌入。

### 使用方法

```python
from torch_rechub.models.ranking import DeepFM

# 创建模型
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | 必填 |
| fm_features | list | FM 部分使用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数 | 必填 |

## 3. DCN

DCN（Deep & Cross Network）在每层交叉网络里重新引入原始特征，让较深的层继续组合前面学到的交叉项。它把交叉分支和深度分支的输出一起用来预测点击。

### 论文引用

```
Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017.
```

### 核心原理

- **Cross Network**：显式学习高阶特征交叉，每层输出为：
  $$x_{l+1} = x_0 x_l^T w_l + b_l + x_l$$
- **Deep Network**：深度神经网络，捕获非线性特征交互
- **联合训练**：Cross Network 和 Deep Network 并行计算，结果拼接后通过全连接层输出

### 使用方法

```python
from torch_rechub.models.ranking import DCN

# 创建模型
model = DCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | Cross 与 Deep 部分共用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数 | 必填 |
| n_cross_layers | int | Cross Network 的层数 | 必填 |

## 4. DCNv2

DCNv2 把 DCN 交叉层中的向量权重换成矩阵。矩阵能分别调节不同维度之间的交互，也会增加参数量。

### 论文引用

```
Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021.
```

`use_low_rank_mixture=True` 会启用低秩专家以减少大矩阵的开销。`model_structure` 决定只用交叉网络，还是把交叉网络与深度网络串联或并联。

### 使用方法

```python
from torch_rechub.models.ranking import DCNv2

# 创建模型
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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | Cross 与 Deep 部分共用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数 | 必填 |
| n_cross_layers | int | Cross Network 的层数 | 必填 |
| model_structure | str | `crossnet_only` / `stacked` / `parallel` | `parallel` |
| use_low_rank_mixture | bool | 是否使用低秩专家混合 CrossNet | True |
| low_rank | int | 低秩维度 | 32 |
| num_experts | int | CrossNetMix 专家数 | 4 |

## 5. EDCN

EDCN 在交叉网络和深度网络之间增加了桥接层。每一层都能交换两个分支学到的特征，最后把两个分支和桥接结果一起用于打分。

### 论文引用

```
Ma, Xiao, et al. "Enhanced Deep & Cross Network for Feature Cross Learning in Click-Through Rate Prediction." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

`bridge_type` 选择两个分支的结合方式。调节模块为两个分支分别学习字段权重。

### 使用方法

```python
from torch_rechub.models.ranking import EDCN

# 创建模型
model = EDCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    bridge_type="hadamard_product",
    use_regulation_module=True,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 模型使用的全部特征 | 必填 |
| mlp_params | dict | 深度神经网络参数 | 必填 |
| n_cross_layers | int | Cross Network 的层数 | 必填 |
| bridge_type | str | Cross/Deep 桥接方式 | `hadamard_product` |
| use_regulation_module | bool | 是否使用特征调节模块 | True |

## 6. AFM

AFM 给每一对字段的交互分配一个权重。例如，模型可以给“时段 × 影片类型”和“城市 × 时段”分配不同的权重。

### 论文引用

```
Xiao, Jun, et al. "Attentional factorization machines: Learning the weight of feature interactions via attention networks." arXiv preprint arXiv:1708.04617 (2017).
```

先对两两字段的嵌入逐元素相乘，再用注意力对这些交互向量加权求和。

### 使用方法

```python
from torch_rechub.models.ranking import AFM

# 创建模型
model = AFM(
    fm_features=sparse_features,
    embed_dim=16,  # 必须与 fm_features 的 embedding 维度一致
    t=64,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| fm_features | list | 参与 FM 交互的稀疏特征，嵌入维度需一致 | 必填 |
| embed_dim | int | 特征嵌入维度 | 必填 |
| t | int | 注意力隐藏层维度 | 64 |

## 7. FiBiNET

FiBiNET 先根据当前样本调整各字段的权重，再用双线性变换计算两两交互。这样，同一个字段与不同字段组合时可以有不同的作用。

### 论文引用

```
Juan, Yuchin, et al. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

模型同时计算原始嵌入和加权嵌入的双线性交互，把两组结果用于最终打分。

### 使用方法

```python
from torch_rechub.models.ranking import FiBiNet

# 创建模型
model = FiBiNet(
    features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    reduction_ratio=3,
    bilinear_type="field_interaction",
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 参与 SENET 和双线性交互的特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| reduction_ratio | int | SENET 压缩比 | 3 |
| bilinear_type | str | `field_all` / `field_each` / `field_interaction` | `field_interaction` |

## 8. DeepFFM

DeepFFM 把字段身份也纳入交互。例如“城市”与“年龄”交互时使用的向量，可以不同于“城市”与“影片类型”交互时的向量。

### 论文引用

```
Xiao, Jun, et al. "Deep learning over multi-field categorical data." European conference on information retrieval. Springer, Cham, 2016.
```

字段多时要留意嵌入表的参数量。这一节也给出带字段注意力的 `FatDeepFFM` 构造方式。

### 使用方法

```python
from torch_rechub.models.ranking import DeepFFM, FatDeepFFM

# FFM 会为每个字段对使用不同 embedding，cross feature 的词表需乘以字段数
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

# 创建 FatDeepFFM 模型（增强版本）
fat_model = FatDeepFFM(
    linear_features=ffm_linear_features,
    cross_features=ffm_cross_features,
    embed_dim=10,
    reduction_ratio=1,
    mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| linear_features | list | 一阶线性部分特征，通常 `embed_dim=1` | 必填 |
| cross_features | list | FFM 交互特征，词表需预留字段偏移空间 | 必填 |
| embed_dim | int | FFM embedding 维度 | 必填 |
| reduction_ratio | int | 仅 FatDeepFFM 需要的 CEN 压缩比 | 必填（FatDeepFFM） |
| mlp_params | dict | FFM 交互后的 MLP 参数 | 必填 |

## 9. BST

BST 将候选物品接在用户的历史序列后面，用 Transformer 同时编码历史和候选，再给这条曝光打分。

### 论文引用

```
Chen, Qiwei, et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba." arXiv preprint arXiv:1905.06874 (2019).
```

### 核心原理

- **Transformer Encoder**：使用多头自注意力机制捕获序列中的依赖关系
- **位置编码**：添加位置信息，保留序列的顺序信息
- **特征融合**：将序列特征与其他特征融合，得到最终的预测结果

### 使用方法

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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 用户画像/上下文特征，不含历史和目标 | 必填 |
| history_features | list | `pooling="concat"` 的历史序列 | 必填 |
| target_features | list | 与历史特征对应的目标物品特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| nhead | int | 多头注意力头数，需整除历史特征维度总和 | 8 |
| dropout | float | Transformer dropout | 0.2 |
| num_layers | int | Transformer Encoder 层数 | 1 |
| max_seq_len | int | 历史长度 + 1 个 target 的上限 | 51 |

## 10. DIN

DIN 用当前候选物品为用户的历史行为分配权重。即使观看历史完全一样，候选是喜剧和候选是科幻片时，也可以得到不同的用户兴趣表示。

### 论文引用

```
Zhou, Guorui, et al. "Deep interest network for click-through rate prediction." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

`target_features` 放当前候选，`history_features` 放对应的行为序列。两边 ID 表示同一类物品时，可用 `shared_with` 让它们共用嵌入表。

### 使用方法

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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 用户画像/上下文特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史序列 | 必填 |
| target_features | list | 与历史特征数量、顺序和维度对应的目标特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| attention_mlp_params | dict | Activation Unit 参数 | 必填 |

## 11. DIEN

DIEN 先用 GRU 读取用户历史，再根据候选物品判断应该保留哪些兴趣变化。训练还需要一条负样本历史，用来监督每一步的兴趣表示。

### 论文引用

```
Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI conference on artificial intelligence. 2019.
```

### 核心原理

- **兴趣抽取层**：GRU 对行为序列建模，辅助损失用正负样本对监督每步隐状态（论文 Eq.7）
- **兴趣演化层**：AUGRU 将注意力分数嵌入更新门，对全序列 softmax 归一化后逐步演化（论文 Eq.14-16）
- **辅助损失**：$L_{aux} = -\frac{1}{N}\sum[\log\sigma(h_t \cdot e^+_{t+1}) + \log(1-\sigma(h_t \cdot e^-_{t+1}))]$
- **Padding 处理**：padding 位（index=0）不参与 GRU/AUGRU 计算；空历史样本保持零隐状态

### 使用方法

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIEN

# target_features 必须设 padding_idx=0，因为 history/neg_history 共享其 embedding 表
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items+1, embed_dim=8, padding_idx=0),
]
# history/neg_history 通过 shared_with 指向 target feature（embed_dict 的 root key）
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
# CTRTrainer 需设 loss_mode=False，因为 forward 返回 (prediction, aux_loss)
```

### 参数说明

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| features | list | 用户画像 / 上下文特征，输入顶层 MLP |
| history_features | list | 正样本行为序列，pooling="concat"，padding_idx=0，shared_with=target_feature |
| neg_history_features | list | 负采样行为序列，同上，shared_with 必须指向 target feature（非 history feature） |
| target_features | list | 目标物品特征，padding_idx=0，用于 AUGRU 注意力 |
| mlp_params | dict | 顶层 MLP 参数，activation 固定为 dice |
| alpha | float | 辅助损失权重，默认 0.2 |

## 12. AutoInt

AutoInt 对一行样本中的不同字段做自注意力。例如，影片类型会结合用户年龄和观看时段等字段重新计算自己的表示。

### 论文引用

```
Song, Weiping, et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

数值字段通过线性变换映射到嵌入空间，与离散字段在同一维度做多头注意力。

### 使用方法

```python
from torch_rechub.models.ranking import AutoInt

# 创建模型
model = AutoInt(
    sparse_features=sparse_features,
    dense_features=dense_features,
    num_layers=3,
    num_heads=2,
    dropout=0.2,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| sparse_features | list | 至少一个、且 `embed_dim` 相同的稀疏特征 | 必填 |
| dense_features | list | 连续特征，可为空列表 | 必填 |
| num_layers | int | Interacting Layer 层数 | 3 |
| num_heads | int | 注意力头数 | 2 |
| dropout | float | 注意力 dropout | 0.0 |
| mlp_params | dict or None | 可选 Deep 分支参数 | None |

## 13. 模型比较

选择排序模型时，先看一次曝光能取到哪些字段。有用户、物品、上下文等表格特征，可以用 WideDeep 或 DeepFM 建立第一组基线，再用 DCN 对比显式交叉的效果。有按时间排列的用户行为，还可以比较 DIN、DIEN 和 BST。三者都用候选物品和历史物品，DIEN 训练时还需要每个历史位置的负例。

比较两个模型时，先固定训练集、验证集和特征编码。用同一批验证记录算 AUC 和 LogLoss，才看得出换模型带来的差异。[怎样读排序指标](/zh/blog/rank)

## 14. 运行一个 Criteo 基线

在[快速入门](/zh/guide/quick_start)中可以看到从原始数据切分、特征编码到 DeepFM 训练和评估的完整代码。从仓库根目录快速验证训练脚本：

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 1 --device cpu
```

## 15. 训练时常见的问题

**为什么训练损失在降，验证 AUC 却开始下降？** 两者衡量的对象不同。先看较早轮次在同一份验证集上的 AUC。`CTRTrainer.fit` 收到验证集时会保存验证 AUC 最好的一轮，可以直接用这组权重预测。

**怎样使用 GPU？** 单卡可给 `CTRTrainer` 传 `device="cuda:0"`。单机多卡可通过 `gpus=[0, 1]` 使用 PyTorch 的 `DataParallel`。批大小要结合每张卡的可用显存调整。
