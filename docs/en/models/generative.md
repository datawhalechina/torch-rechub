---
title: Generative Recommendation Models
description: Torch-RecHub generative recommendation models detailed introduction
---

# Generative Recommendation Models

Generative recommendation models are an emerging approach that leverages generative AI technologies (such as large language models) for recommendations, capable of generating personalized recommendation content and providing richer, more natural recommendation experiences. Torch-RecHub provides various advanced generative recommendation models that combine the advantages of recommendation systems and generative AI.

## 1. HSTUModel

### Description

HSTU (Hierarchical Sequence Transformer Unit) is a hierarchical sequence transformation unit designed for large-scale sequence recommendation, capable of supporting trillion-parameter recommendation systems.

### Core Principles

- **Hierarchical Structure**: Uses hierarchical design to decompose long sequences into multiple sub-sequences, improving model parallelism and scalability
- **Transformer Architecture**: Based on Transformer architecture, capable of capturing long-range dependencies
- **Large-scale Pretraining**: Supports large-scale pretraining, learning universal representations from massive data
- **Efficient Inference**: Optimized inference process, supporting real-time recommendations

### Usage

```python
from torch_rechub.models.generative import HSTUModel
from torch_rechub.basic.features import SparseFeature, SequenceFeature

# Define features
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling="mean")
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# Create model
model = HSTUModel(
    user_features=user_features,
    item_features=item_features,
    transformer_params={
        "num_layers": 2,
        "num_heads": 4,
        "hidden_size": 128,
        "intermediate_size": 256,
        "dropout": 0.2
    },
    hierarchical_params={
        "level1_window_size": 10,
        "level2_window_size": 5
    }
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| transformer_params | dict | Transformer parameters | None |
| hierarchical_params | dict | Hierarchical structure parameters | None |

### Use Cases

- Large-scale sequence recommendation
- Long sequence modeling
- Real-time recommendation scenarios
- Trillion-parameter recommendation systems

## 2. HLLMModel

### Description

HLLM (Hybrid Large Language Model) is a hybrid recommendation model that integrates large language model (LLM) capabilities, combining the collaborative filtering ability of recommendation systems with the semantic understanding ability of LLMs.

### Core Principles

- **Hybrid Architecture**: Combines the advantages of traditional recommendation models and large language models
- **Semantic Understanding**: Leverages LLM's powerful semantic understanding ability to process text information
- **Collaborative Filtering**: Retains traditional recommendation model's collaborative filtering ability, utilizing user-item interaction data
- **Multi-modal Fusion**: Supports fusion of multiple modalities such as text and images

### Usage

```python
from torch_rechub.models.generative import HLLMModel

# Create model
model = HLLMModel(
    user_features=user_features,
    item_features=item_features,
    llm_params={
        "model_name": "bert-base-uncased",
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1
    },
    fusion_params={
        "fusion_type": "concat",
        "fusion_dims": [512, 256, 128],
        "dropout": 0.2
    }
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| llm_params | dict | Large language model parameters | None |
| fusion_params | dict | Feature fusion parameters | None |

### Use Cases

- Recommendation scenarios integrating LLM capabilities
- Recommendation scenarios with rich text information
- Scenarios requiring generative recommendations
- Multi-modal recommendation scenarios

## 3. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Use Cases |
| --- | --- | --- | --- | --- |
| HSTUModel | High | High | Medium | Large-scale sequence recommendation, long sequence modeling |
| HLLMModel | High | High | Low | LLM integration, text-rich scenarios |

## 4. Usage Recommendations

1. **Choose models based on business requirements**:
   - For large-scale sequence recommendation, use HSTUModel
   - For scenarios requiring LLM capabilities, use HLLMModel
   - For text-rich recommendation scenarios, use HLLMModel

2. **Choose models based on computational resources**:
   - With limited resources, use HSTUModel
   - With sufficient resources, try HLLMModel
   - Consider model compression techniques like knowledge distillation, quantization, etc.

3. **Model training recommendations**:
   - Use pretrain + finetune approach to improve model effectiveness and training efficiency
   - Use mixed precision training to accelerate model training
   - Use distributed training to handle large-scale data

4. **Model deployment recommendations**:
   - Consider model compression techniques to reduce model size and inference latency
   - Use service-oriented deployment to support high-concurrency requests
   - Consider edge computing to deploy models on edge devices

## 5. Complete Training Example

```python
from torch_rechub.models.generative import HSTUModel
from torch_rechub.trainers import GenRecTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import SparseFeature, SequenceFeature

# 1. Define features
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling="mean")
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 2. Prepare data
# Assume x and y are preprocessed feature and label data
x = {
    "user_id": user_id_data,
    "user_history": user_history_data,
    "item_id": item_id_data,
    "category": category_data
}
y = label_data  # click/no-click labels

# 3. Create data generator
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 4. Create model
model = HSTUModel(
    user_features=user_features,
    item_features=item_features,
    transformer_params={
        "num_layers": 2,
        "num_heads": 4,
        "hidden_size": 128,
        "intermediate_size": 256,
        "dropout": 0.2
    },
    hierarchical_params={
        "level1_window_size": 10,
        "level2_window_size": 5
    }
)

# 5. Create trainer
trainer = GenRecTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/hstu"
)

# 6. Train model
trainer.fit(train_dl, val_dl)

# 7. Evaluate model
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc}")

# 8. Export model
trainer.export_onnx("hstu.onnx")

# 9. Model prediction
preds = trainer.predict(trainer.model, test_dl)
print(f"Predictions shape: {preds.shape}")
```

## 6. FAQ

### Q: How to handle large-scale data?
A: Try the following approaches:
- Use distributed training with multi-GPU or multi-machine parallel training
- Use data sampling techniques like negative sampling, stratified sampling, etc.
- Use model parallelism or pipeline parallelism for very large models
- Consider mixed precision training to accelerate training

### Q: How to improve inference efficiency of generative recommendation models?
A: Try the following approaches:
- Use model compression techniques like knowledge distillation, quantization, pruning, etc.
- Use deployment optimization like TensorRT, ONNX Runtime, etc.
- Consider edge computing to deploy models on edge devices
- Use asynchronous inference or batch processing to improve concurrency

### Q: How to evaluate generative recommendation model performance?
A: Try the following approaches:
- Traditional recommendation metrics: AUC, Precision@K, Recall@K, NDCG@K, etc.
- Generative evaluation metrics: BLEU, ROUGE, METEOR, Perplexity, etc.
- Human evaluation: User surveys or A/B testing
- Business metrics: CTR, conversion rate, user retention, etc.

### Q: How to handle cold start problems?
A: Try the following approaches:
- For new users, use content-based or popularity-based recommendations
- For new items, leverage LLM's semantic understanding based on item descriptions
- Use transfer learning to transfer knowledge from related domains
- Use meta-learning to quickly adapt to new users or items

## 7. Application Scenarios

1. **Personalized Content Generation**:
   - Generate personalized recommendation reasons
   - Generate personalized product descriptions
   - Generate personalized marketing copy

2. **Multi-modal Recommendation**:
   - Combine text, image, audio, and other modalities
   - Generate multi-modal recommendation content
   - Support cross-modal recommendations

3. **Interactive Recommendation**:
   - Support natural language interaction between users and recommendation systems
   - Dynamically adjust recommendations based on user feedback
   - Generate conversational recommendations

4. **Contextual Recommendation**:
   - Generate recommendations based on user's current context
   - Generate context-aware recommendation content
   - Support complex scenario recommendations

## 8. Future Trends

1. **Deep Integration of LLMs and Recommendation Systems**:
   - More tightly combine the advantages of LLMs and recommendation systems
   - Develop LLMs specifically optimized for recommendation scenarios
   - Leverage LLM's contextual understanding for more personalized recommendations

2. **Multi-modal Generative Recommendation**:
   - Combine multiple modalities for generative recommendation
   - Support cross-modal content generation and recommendation
   - Develop more efficient multi-modal fusion methods

3. **Real-time Generative Recommendation**:
   - Achieve low-latency generative recommendation
   - Support real-time user interaction and feedback
   - Develop more efficient inference architectures

4. **Controllable Generative Recommendation**:
   - Support user control and adjustment of recommendation results
   - Achieve explainable and trustworthy generative recommendation
   - Develop safer and more reliable generative recommendation systems

5. **Large-scale Generative Recommendation**:
   - Support billions of users and items for large-scale recommendation
   - Develop more efficient model training and inference methods
   - Achieve distributed and scalable generative recommendation systems

Generative recommendation is an important development direction for recommendation systems, capable of providing richer, more natural, and more personalized recommendation experiences. Torch-RecHub provides various advanced generative recommendation models for developers to choose based on business requirements. With the continuous development of large language models and generative AI technologies, generative recommendation will be applied in more scenarios, providing users with better recommendation experiences.