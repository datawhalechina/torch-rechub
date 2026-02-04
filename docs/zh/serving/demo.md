---
title: 在线服务示例
description: Torch-RecHub 在线服务部署示例
---

# 在线服务示例

本文档提供 Torch-RecHub 模型在线服务部署的完整示例。

## 排序模型部署

### 1. 导出 ONNX 模型

```python
from torch_rechub.trainers import CTRTrainer

# 训练完成后导出
trainer.export_onnx("deepfm.onnx", dynamic_batch=True)
```

### 2. ONNX Runtime 推理

```python
import numpy as np
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession("deepfm.onnx")

# 准备输入
inputs = {
    "city": np.array([[1, 2, 3]], dtype=np.int64),
    "age": np.array([[0.5, 0.3, 0.8]], dtype=np.float32),
}

# 推理
outputs = session.run(None, inputs)
predictions = outputs[0]
```

## 召回模型部署

### 1. 导出双塔模型

```python
from torch_rechub.trainers import MatchTrainer

# 分别导出用户塔和物品塔
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

### 2. 离线构建物品索引

```python
import numpy as np
import onnxruntime as ort

# 加载物品塔
item_session = ort.InferenceSession("item_tower.onnx")

# 计算所有物品的 embedding
item_embeddings = []
for batch in item_dataloader:
    inputs = {k: v.numpy() for k, v in batch.items()}
    emb = item_session.run(None, inputs)[0]
    item_embeddings.append(emb)

item_embeddings = np.concatenate(item_embeddings)
```

### 3. 在线召回服务（Milvus）

```python
import torch
import numpy as np
import onnxruntime as ort
from torch_rechub.serving import builder_factory

# 加载用户塔
user_session = ort.InferenceSession("user_tower.onnx")

# 连接 Milvus 服务
embed_dim = 64  # embedding 维度
builder = builder_factory(
    "milvus",
    d=embed_dim,
    index_type="HNSW",
    metric="IP",
    host="localhost",
    port=19530
)

# 将物品 embedding 写入 Milvus
with builder.from_embeddings(torch.tensor(item_embeddings)) as indexer:
    # 计算用户 embedding
    user_inputs = {"user_id": np.array([[123]])}
    user_emb = user_session.run(None, user_inputs)[0]

    # 向量检索
    ids, scores = indexer.query(torch.tensor(user_emb), top_k=100)
    recall_items = ids[0].tolist()
```

## 最佳实践

1. **模型量化**：使用 INT8/FP16 量化降低推理延迟
2. **批量推理**：合并请求进行批量推理提高吞吐
3. **索引预热**：服务启动时预加载索引到内存
4. **监控告警**：监控推理延迟和错误率

