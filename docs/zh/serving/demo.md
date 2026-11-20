---
title: 用训练好的模型做预测
description: 核对 DeepFM 的 ONNX 分数，用 DSSM 和 Annoy 查找电影 ID
---

# 从训练结果到一次预测

下面各用一条输入检查训练结果。刚跑完 DeepFM，可以先核对导出前后的点击分数。刚跑完 DSSM，可以试着给一个用户找电影。每段代码都在对应的快速入门示例所在的 Python 会话里运行，分别需要 `onnx` 和 `annoy` 可选依赖。[快速入门](/zh/guide/quick_start)

## 用同一条 Criteo 记录校验排序分数

从留出的 `test` 表取第一条原始记录，只调用一次 `encode_features`，把同一份编码后的特征分别交给两个模型。`example_batch` 则取自实际的测试批次，让导出器看到和训练时一样的字段形状。

```python
import numpy as np
import onnxruntime as ort
import torch

example_batch, _ = next(iter(test_dl))
trainer.export_onnx("deepfm.onnx", dummy_input=example_batch)
x = encode_features(test.iloc[:1])
trainer.model.eval()
with torch.no_grad():
    score = trainer.model({name: torch.from_numpy(v) for name, v in x.items()}).cpu().numpy()

session = ort.InferenceSession("deepfm.onnx", providers=["CPUExecutionProvider"])
inputs = {field.name: x[field.name] for field in session.get_inputs()}
onnx_score = session.run(None, inputs)[0]
np.testing.assert_allclose(onnx_score, score, rtol=1e-4, atol=1e-6)
print("点击分数", onnx_score.tolist())
```

能打印出“点击分数”，说明这条记录的两次预测在代码给定的误差范围内一致。遇到断言失败时，先核对字段名、形状和类型。类别 ID 应为 `int64`，连续字段为 `float32`。

部署时把 `vocabs`、`scaler` 和连续字段的顺序 `dense_features` 随模型一起保存，新请求仍用训练时的 `encode_features` 编码。如果之后量化了模型，再用整份留出数据比较 AUC 和 LogLoss。

## 从用户向量查回原始电影 ID

`item_embedding` 的第 0 行是 `all_item` 中的第一部电影。如果 Annoy 返回行号 0，应该推荐的是这部电影。下面用 `movies.ann` 存索引，再用 `movie_ids.npy` 存一份同顺序的原始电影 ID。

```python
import numpy as np
from torch_rechub.serving import builder_factory

builder = builder_factory("annoy", d=item_embedding.shape[1], metric="angular")
row_to_movie = raw_movie_ids[all_item["movie_id"] - 1]
np.save("movie_ids.npy", row_to_movie)
with builder.from_embeddings(item_embedding.cpu()) as index:
    index.save("movies.ann")

row_to_movie = np.load("movie_ids.npy")
with builder.from_index_file("movies.ann") as index:
    rows, distances = index.query(user_embedding[:1].cpu(), top_k=min(10, len(row_to_movie)))
movie_ids = row_to_movie[rows[0].numpy()]
print("推荐电影", movie_ids.tolist())
```

代码最后打印的是原始电影 ID，可直接用来查片名。用新的用户请求试时，沿用训练时的词表和历史序列长度，用同一版用户塔生成查询向量。换了物品塔权重或电影排列顺序，记得一起更新 `movies.ann` 和 `movie_ids.npy`。[双塔导出与量化](/zh/serving/onnx) · [向量索引](/zh/serving/vector_index)

