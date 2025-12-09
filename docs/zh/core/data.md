---
title: 数据流水线
description: Torch-RecHub 数据加载与预处理
---

# 数据流水线

Torch-RecHub提供了完整的数据处理流水线，包括数据集类、数据生成器和工具函数，用于处理推荐系统中的各种数据需求。

## 数据类

### TorchDataset

用于训练和验证的数据集合，包含特征和标签。

```python
from torch_rechub.utils.data import TorchDataset

# 创建数据集
dataset = TorchDataset(x, y)
```

**参数说明：**
- `x`：特征字典，键为特征名称，值为特征数据
- `y`：标签数据

### PredictDataset

用于预测的数据集合，仅包含特征。

```python
from torch_rechub.utils.data import PredictDataset

# 创建预测数据集
dataset = PredictDataset(x)
```

**参数说明：**
- `x`：特征字典，键为特征名称，值为特征数据

## 数据生成器

### DataGenerator

用于生成排序模型和多任务模型的数据加载器。

```python
from torch_rechub.utils.data import DataGenerator

# 创建数据生成器
dg = DataGenerator(x, y)
# 生成数据加载器
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],  # 训练集:验证集:测试集比例
    batch_size=256,           # 批次大小
    num_workers=8             # 并行工作线程数
)
```

**参数说明：**
- `x`：特征数据
- `y`：标签数据

**generate_dataloader方法参数：**
- `split_ratio`：数据分割比例，长度为2
- `batch_size`：批次大小
- `num_workers`：并行工作线程数

### MatchDataGenerator

用于生成召回模型的数据加载器。

```python
from torch_rechub.utils.data import MatchDataGenerator

# 创建召回数据生成器
dg = MatchDataGenerator(x, y)
# 生成数据加载器
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test_user=x_test_user,  # 测试用户数据
    x_all_item=x_all_item,    # 所有物品数据
    batch_size=256,           # 批次大小
    num_workers=8             # 并行工作线程数
)
```

**参数说明：**
- `x`：特征数据
- `y`：标签数据，可选

**generate_dataloader方法参数：**
- `x_test_user`：测试用户数据
- `x_all_item`：所有物品数据
- `batch_size`：批次大小
- `num_workers`：并行工作线程数

## 工具函数

### get_auto_embedding_dim

根据类别数量自动计算嵌入向量长度。

```python
from torch_rechub.utils.data import get_auto_embedding_dim

# 自动计算嵌入向量长度
embed_dim = get_auto_embedding_dim(vocab_size=1000)
```

**参数说明：**
- `num_classes`：类别数量

**返回值：**
- 嵌入向量长度，计算公式：`int(np.floor(6 * np.pow(num_classes, 0.25)))`

### get_loss_func

根据任务类型获取对应的损失函数。

```python
from torch_rechub.utils.data import get_loss_func

# 获取分类任务损失函数
loss_func = get_loss_func(task_type="classification")
# 获取回归任务损失函数
loss_func = get_loss_func(task_type="regression")
```

**参数说明：**
- `task_type`：任务类型，可选值：classification（分类）、regression（回归）

**返回值：**
- 对应的损失函数实例

## 数据处理流程

1. **特征定义**：使用DenseFeature、SparseFeature、SequenceFeature定义特征
2. **数据加载**：加载原始数据
3. **特征编码**：对类别型特征进行LabelEncoder编码
4. **序列处理**：对序列特征进行填充、截断等处理
5. **样本构造**：构造训练样本，包括负采样等
6. **数据生成**：使用DataGenerator或MatchDataGenerator生成数据加载器
7. **模型训练**：将数据加载器传入模型进行训练