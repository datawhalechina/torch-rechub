---
title: Make a prediction with a trained model
description: Check DeepFM scores after ONNX export and retrieve movie IDs with DSSM and Annoy
---

# Make a prediction with a trained model

If you have just trained DeepFM, check whether exporting it changes a click score. If you trained DSSM, try finding movies for one user. Run each example in the Python session where you ran the corresponding Quick Start, with the `onnx` or `annoy` extra installed. [Quick Start](/guide/quick_start)

## Compare ranking scores for one Criteo record

Take the first raw record from the held-out `test` table. Call `encode_features` once and give both models the same encoded inputs. Use a real test batch as `example_batch` so the exported model sees the same field shapes as the training model.

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
print("Click score", onnx_score.tolist())
```

If the code prints the click score, the two predictions for this record agree within the specified tolerance. If the assertion fails, check the input names, shapes and types first. Category IDs should be `int64` and continuous features `float32`.

For deployment, save `vocabs`, `scaler` and the order of the continuous fields in `dense_features` alongside the model. Encode each new request with the same `encode_features` function. If you later quantize the model, compare AUC and LogLoss on the full held-out set.

## Retrieve original movie IDs from a user vector

Row 0 of `item_embedding` represents the first movie in `all_item`. If Annoy returns row 0, that is the movie to recommend. Save the index in `movies.ann` and the original movie IDs in matching row order in `movie_ids.npy`.

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
print("Recommended movies", movie_ids.tolist())
```

The code prints original movie IDs that you can use to look up titles. For a new request, use the training vocabulary, history length and corresponding user tower to generate the query vector. If you change the item tower weights or the movie order, rebuild `movies.ann` and `movie_ids.npy` together. [Export the two towers](/serving/onnx) · [Vector indexes](/serving/vector_index)
