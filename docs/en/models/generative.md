---
title: Generative Recommendation Models
description: Torch-RecHub generative recommendation models
---

# Generative Recommendation Models

Generative recommendation leverages generative AI (e.g., LLMs) to produce richer, more natural personalized recommendations. Torch-RecHub includes advanced models that combine RecSys and generative capabilities.

## 1. HSTUModel

Hierarchical Sequence Transformer Unit for large-scale sequence recommendation.

- **Hierarchy**: splits long sequences for parallelism and scalability.  
- **Transformer**: captures long-range dependencies.  
- **Pretraining-friendly**: suited for large-scale pretraining.  
- **Efficient inference**: optimized for real-time serving.

```python
from torch_rechub.models.generative import HSTUModel
from torch_rechub.basic.features import SparseFeature, SequenceFeature

user_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=32),
    SequenceFeature("user_history", vocab_size=100000, embed_dim=32, pooling="mean"),
]

item_features = [
    SparseFeature("item_id", vocab_size=100000, embed_dim=32),
    SparseFeature("category", vocab_size=1000, embed_dim=16),
]

model = HSTUModel(
    user_features=user_features,
    item_features=item_features,
    transformer_params={"num_layers": 2, "num_heads": 4, "hidden_size": 128, "intermediate_size": 256, "dropout": 0.2},
    hierarchical_params={"level1_window_size": 10, "level2_window_size": 5},
)
```

## 2. HLLMModel

Hybrid LLM-based model combining LLM semantic strength with collaborative signals.

- **Hybrid architecture**: blends classic recommendation and LLM.  
- **Semantic understanding**: strong text comprehension.  
- **Multi-modal friendly**: supports text/image fusion.  
- **Flexible fusion**: configurable fusion MLP.

```python
from torch_rechub.models.generative import HLLMModel

model = HLLMModel(
    user_features=user_features,
    item_features=item_features,
    llm_params={
        "model_name": "bert-base-uncased",
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1,
    },
    fusion_params={
        "fusion_type": "concat",
        "fusion_dims": [512, 256, 128],
        "dropout": 0.2,
    },
)
```

## 3. Model Comparison (high level)

| Model     | Complexity | Capacity | Efficiency | Best for |
| ---       | ---        | ---      | ---        | --- |
| HSTUModel | High       | High     | Medium     | Large-scale, long sequences |
| HLLMModel | High       | High     | Lower      | LLM-enhanced, text/multi-modal |

## 4. Practical Tips

- Choose by need: large-scale seq → HSTU; need LLM/semantics → HLLM.  
- Resource-aware: try HSTU when resources are tight; HLLM when plentiful (or after compression).  
- Training: pretrain + finetune; use mixed precision; consider distributed training.  
- Deployment: consider distillation/quantization; serve with ONNX Runtime/TensorRT; edge offloading if needed.

## 5. Example Workflow (HSTU)

```python
from torch_rechub.models.generative import HSTUModel
from torch_rechub.trainers import GenRecTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import SparseFeature, SequenceFeature

# 1) Define features (user_features, item_features) as above
# 2) Prepare data dicts x, labels y
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 3) Create model
model = HSTUModel(user_features=user_features, item_features=item_features, transformer_params={...}, hierarchical_params={...})

# 4) Trainer
trainer = GenRecTrainer(model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-4}, n_epoch=50, earlystop_patience=10, device="cuda:0", model_path="saved/hstu")

# 5) Train / Eval / Export / Predict
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
trainer.export_onnx("hstu.onnx")
preds = trainer.predict(trainer.model, test_dl)
```

## 6. FAQs (concise)

- **Large-scale data?** Use distributed training, negative sampling, mixed precision.  
- **Speeding up inference?** Distill/quantize/prune; ONNX Runtime/TensorRT; batch/async inference.  
- **Evaluation?** AUC/Precision@K/Recall@K/NDCG@K; for generation: BLEU/ROUGE/METEOR/Perplexity; also business KPIs and A/B tests.  
- **Cold start?** Content-based or popularity for new users; LLM text semantics for new items; transfer learning/meta-learning.

## 7. Application Scenarios

- Personalized content generation (reasons, descriptions, copy).  
- Multi-modal recommendation (text/image/audio).  
- Conversational/interactive recommendation.  
- Contextual/scene-aware recommendation.

## 8. Outlook

LLM-rec convergence, multi-modal generation, real-time/low-latency generation, controllable and explainable generation, and scalable (billions of users/items) training/inference.

