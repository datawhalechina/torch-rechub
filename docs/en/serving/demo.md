---
title: Online Service Demo
description: Torch-RecHub online service deployment examples
---

# Online Service Demo

This document provides complete examples for deploying Torch-RecHub models as online services.

## Ranking Model Deployment

### 1. Export ONNX Model

```python
from torch_rechub.trainers import CTRTrainer

# Export after training
trainer.export_onnx("deepfm.onnx", dynamic_batch=True)
```

### 2. ONNX Runtime Inference

```python
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("deepfm.onnx")

# Prepare inputs
inputs = {
    "city": np.array([[1, 2, 3]], dtype=np.int64),
    "age": np.array([[0.5, 0.3, 0.8]], dtype=np.float32),
}

# Inference
outputs = session.run(None, inputs)
predictions = outputs[0]
```

## Matching Model Deployment

### 1. Export Two-Tower Model

```python
from torch_rechub.trainers import MatchTrainer

# Export user tower and item tower separately
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

### 2. Offline Item Index Building

```python
import numpy as np
import onnxruntime as ort

# Load item tower
item_session = ort.InferenceSession("item_tower.onnx")

# Compute embeddings for all items
item_embeddings = []
for batch in item_dataloader:
    inputs = {k: v.numpy() for k, v in batch.items()}
    emb = item_session.run(None, inputs)[0]
    item_embeddings.append(emb)

item_embeddings = np.concatenate(item_embeddings)
```

### 3. Online Retrieval Service (Milvus)

```python
import torch
import numpy as np
import onnxruntime as ort
from torch_rechub.serving import builder_factory

# Load user tower
user_session = ort.InferenceSession("user_tower.onnx")

# Connect to Milvus service
embed_dim = 64  # embedding dimension
builder = builder_factory(
    "milvus",
    d=embed_dim,
    index_type="HNSW",
    metric="IP",
    host="localhost",
    port=19530
)

# Write item embeddings to Milvus
with builder.from_embeddings(torch.tensor(item_embeddings)) as indexer:
    # Compute user embedding
    user_inputs = {"user_id": np.array([[123]])}
    user_emb = user_session.run(None, user_inputs)[0]

    # Vector retrieval
    ids, scores = indexer.query(torch.tensor(user_emb), top_k=100)
    recall_items = ids[0].tolist()
```

## Best Practices

1. **Model Quantization**: Use INT8/FP16 quantization to reduce inference latency
2. **Batch Inference**: Combine requests for batch inference to improve throughput
3. **Index Preloading**: Preload indexes into memory when service starts
4. **Monitoring & Alerting**: Monitor inference latency and error rates
