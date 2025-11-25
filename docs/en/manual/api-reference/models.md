---
title: Models API Reference
description: Complete API documentation for all models including recall, ranking, and multi-task learning models
---

# Models API Reference

This section provides detailed API documentation for all models in Torch-RecHub.

## Recall Models

Recall models are primarily used in the recall stage for quick retrieval of relevant items from massive candidate sets. They typically adopt two-tower or sequential model structures to meet the efficiency requirements of the recall stage.

### Two-Tower Model Series

#### DSSM (Deep Structured Semantic Model)
- **Introduction**: Originally proposed by Microsoft for semantic matching and later widely applied in recommender systems. Adopts classic two-tower structure that separately represents users and items, computing similarity through inner product. This structure allows pre-computation of item vectors during online serving, greatly improving service efficiency. The key lies in learning effective user and item representations.
- **Parameters**:
  - `user_features` (list): List of user features
  - `item_features` (list): List of item features
  - `hidden_units` (list): List of hidden layer units
  - `dropout_rates` (list): List of dropout rates
  - `embedding_dim` (int): Final representation vector dimension

#### Facebook DSSM
- **Introduction**: Facebook's improved version of DSSM that incorporates multi-task learning framework. Besides the main recall task, it adds auxiliary tasks to help learn better feature representations. The model can simultaneously optimize multiple related objectives like clicks, favorites, purchases, etc., learning richer user and item representations.
- **Parameters**:
  - `user_features` (list): List of user features
  - `item_features` (list): List of item features
  - `hidden_units` (list): List of hidden layer units
  - `num_tasks` (int): Number of tasks
  - `task_types` (list): List of task types

#### YouTube DNN
- **Introduction**: A deep recall model proposed by YouTube, designed for large-scale video recommendation scenarios. The model aggregates user viewing history through average pooling and combines it with other user features. Innovatively introduces negative sampling techniques and multi-task learning framework to improve training efficiency and effectiveness.
- **Parameters**:
  - `user_features` (list): List of user features
  - `item_features` (list): List of item features
  - `hidden_units` (list): List of hidden layer units
  - `embedding_dim` (int): Embedding dimension
  - `max_seq_len` (int): Maximum sequence length

### Sequential Recommendation Series

#### GRU4Rec
- **Introduction**: A pioneering work that first applied GRU networks to session-based sequential recommendation. Through GRU network, it captures temporal dependencies in user behavior sequences, with hidden states at each time step containing information about historical behaviors. The model also introduces special mini-batch construction methods and loss function designs to adapt to the characteristics of sequential recommendation.
- **Parameters**:
  - `item_num` (int): Total number of items
  - `hidden_size` (int): Size of GRU hidden layer
  - `num_layers` (int): Number of GRU layers
  - `dropout_rate` (float): Dropout rate
  - `embedding_dim` (int): Item embedding dimension

#### NARM (Neural Attentive Recommendation Machine)
- **Introduction**: A sequential recommendation model that introduces attention mechanism on top of GRU4Rec. Through attention mechanism, the model can dynamically focus on relevant behaviors in the sequence based on the current prediction target. It maintains both global and local sequence representations, comprehensively capturing user's short-term interests. This design enables better handling of user interest diversity and dynamics.
- **Parameters**:
  - `item_num` (int): Total number of items
  - `hidden_size` (int): Size of hidden layer
  - `attention_size` (int): Size of attention layer
  - `dropout_rate` (float): Dropout rate
  - `embedding_dim` (int): Item embedding dimension

#### SASRec (Self-Attentive Sequential Recommendation)
- **Introduction**: A representative work that applies Transformer structure to sequential recommendation. Through self-attention mechanism, the model can directly compute and learn relationships between any two behaviors in the sequence, unrestricted by RNN's inherent sequential dependencies. Position encoding helps preserve temporal information of behaviors, while multi-layer structure allows the model to extract increasingly abstract behavior patterns layer by layer. Compared to RNN-based models, it offers better parallelism and scalability.
- **Parameters**:
  - `item_num` (int): Total number of items
  - `max_len` (int): Maximum sequence length
  - `num_heads` (int): Number of attention heads
  - `num_layers` (int): Number of Transformer layers
  - `hidden_size` (int): Hidden layer dimension
  - `dropout_rate` (float): Dropout rate

#### MIND (Multi-Interest Network with Dynamic routing)
- **Introduction**: A recall model designed for user's diverse interests. Through capsule network and dynamic routing mechanism, it extracts multiple interest vectors from user's behavior sequence. Each interest vector represents user preferences in different aspects, providing a more comprehensive characterization of user interest distribution.
- **Parameters**:
  - `item_num` (int): Total number of items
  - `num_interests` (int): Number of interest vectors
  - `routing_iterations` (int): Number of dynamic routing iterations
  - `hidden_size` (int): Hidden layer dimension
  - `embedding_dim` (int): Item embedding dimension

## Ranking Models

Ranking models are primarily used in the fine-ranking stage to precisely rank candidate items. They learn complex interactions between users and items through deep learning methods to generate final ranking scores.

### Wide & Deep Series

#### WideDeep
- **Introduction**: A classic model proposed by Google in 2016 that combines the advantages of linear models and deep neural networks. The Wide part performs memorization through feature crosses, suitable for modeling direct, explicit feature correlations; the Deep part performs generalization through deep networks, capable of learning implicit, high-order feature relationships. This combination allows the model to both memorize historical patterns and generalize to new patterns.
- **Parameters**:
  - `wide_features` (list): List of features for the wide part, used in linear layer
  - `deep_features` (list): List of features for the deep part, used in deep network
  - `hidden_units` (list): List of hidden layer units for the deep network, e.g., [256, 128, 64]
  - `dropout_rates` (list): Dropout rates for each layer, used for preventing overfitting

#### DeepFM
- **Introduction**: A model that combines Factorization Machines (FM) feature interactions with deep learning models. The FM part efficiently models second-order feature interactions, while the Deep part learns high-order feature relationships. Compared to Wide&Deep, DeepFM doesn't require manual feature engineering and can automatically learn feature crosses. The model consists of three parts: first-order features, FM's second-order interactions, and deep network's high-order interactions.
- **Parameters**:
  - `features` (list): List of features
  - `hidden_units` (list): Hidden layer units for DNN part
  - `dropout_rates` (list): List of dropout rates
  - `embedding_dim` (int): Feature embedding dimension

#### DCN / DCN-V2
- **Introduction**: Learns feature interactions explicitly through specially designed Cross Network layers. Each cross layer performs interactions between feature vectors and their original form, increasing the degree of feature crossing as the depth increases. DCN-V2 improves the cross network parameterization, offering both "vector" and "matrix" options, maintaining model expressiveness while improving efficiency.
- **Parameters**:
  - `features` (list): List of features
  - `cross_num` (int): Number of cross layers
  - `hidden_units` (list): Hidden layer units for DNN part
  - `cross_parameterization` (str, DCN-V2): Cross parameterization method, "vector" or "matrix"

#### AFM (Attentional Factorization Machine)
- **Introduction**: Introduces attention mechanism to FM, assigning different importance weights to different feature interactions. Through the attention network, it adaptively learns the importance of feature interactions, identifying feature combinations that are more relevant to the prediction target.
- **Parameters**:
  - `features` (list): List of features
  - `attention_units` (list): Hidden layer units for attention network
  - `embedding_dim` (int): Feature embedding dimension
  - `dropout_rate` (float): Dropout rate for attention network

#### FiBiNET (Feature Importance and Bilinear feature Interaction Network)
- **Introduction**: Dynamically learns feature importance through SENET mechanism and uses bilinear layers for feature interaction. The SENET module helps identify important features, while bilinear interaction provides richer feature interaction methods than inner products.
- **Parameters**:
  - `features` (list): List of features
  - `bilinear_type` (str): Bilinear layer type, options: "field_all"/"field_each"/"field_interaction"
  - `hidden_units` (list): Hidden layer units for DNN part
  - `reduction_ratio` (int): Reduction ratio for SENET module

### Attention-based Series

#### DIN (Deep Interest Network)
- **Introduction**: A model designed for user interest diversity, using attention mechanism for adaptive learning of user historical behaviors. The model dynamically calculates relevance weights of user historical behaviors based on the current candidate ad, thereby activating relevant user interests and capturing diverse user preferences. It innovatively introduced attention mechanism to recommender systems, pioneering a new paradigm for behavior sequence modeling.
- **Parameters**:
  - `features` (list): List of base features
  - `behavior_features` (list): List of behavior features for attention calculation
  - `attention_units` (list): Hidden layer units for attention network
  - `hidden_units` (list): Hidden layer units for DNN part
  - `activation` (str): Activation function type

#### DIEN (Deep Interest Evolution Network)
- **Introduction**: An advanced version of DIN that models the dynamic evolution of user interests through interest evolution layer. It uses GRU structure to capture interest evolution and innovatively designs AUGRU (GRU with Attentional Update Gate) to make the interest evolution process aware of target items. It also includes auxiliary loss to supervise the training of interest extraction layer. This design not only captures the dynamic changes of user interests but also models the temporal dependencies of interests.
- **Parameters**:
  - `features` (list): List of base features
  - `behavior_features` (list): List of behavior features
  - `interest_units` (list): Units for interest extraction layer
  - `gru_type` (str): GRU type, "AUGRU" or "AIGRU"
  - `hidden_units` (list): Hidden layer units for DNN part

#### BST (Behavior Sequence Transformer)
- **Introduction**: A pioneering work that introduces Transformer architecture to recommender systems for modeling user behavior sequences. Through self-attention mechanism, the model can directly compute relationships between any two behaviors in the sequence, overcoming the limitations of RNN models in processing long sequences. Position embedding helps the model perceive temporal information of behaviors, while multi-head attention allows the model to understand user behavior patterns from multiple perspectives.
- **Parameters**:
  - `features` (list): List of base features
  - `behavior_features` (list): List of behavior features
  - `num_heads` (int): Number of attention heads
  - `num_layers` (int): Number of Transformer layers
  - `hidden_size` (int): Hidden layer dimension
  - `dropout_rate` (float): Dropout rate

#### EDCN (Enhancing Explicit and Implicit Feature Interactions)
- **Introduction**: A deep cross network that enhances both explicit and implicit feature interactions. Through a newly designed cross network structure, it considers both explicit and implicit feature interactions. Introduces gating mechanism to regulate the importance of different orders of feature interactions and uses residual connections to alleviate training issues in deep networks.
- **Parameters**:
  - `features` (list): List of features
  - `cross_num` (int): Number of cross layers
  - `hidden_units` (list): Hidden layer units for DNN part
  - `gate_type` (str): Gate type, "FGU" or "BGU"

## Multi-task Models

Multi-task models learn multiple related tasks jointly to achieve knowledge sharing and transfer, improving overall model performance.

### SharedBottom
- **Introduction**: The most basic multi-task learning model that shares parameters in bottom network for extracting common feature representations. The shared layers learn common features across tasks, while task-specific layers learn individualized features for each task. This simple yet effective structure laid the foundation for multi-task learning.
- **Parameters**:
  - `features` (list): List of features
  - `hidden_units` (list): Hidden layer units for shared network
  - `task_hidden_units` (list): Hidden layer units for task-specific networks
  - `num_tasks` (int): Number of tasks
  - `task_types` (list): List of task types

### ESMM (Entire Space Multi-Task Model)
- **Introduction**: An innovative multi-task model proposed by Alibaba specifically designed to address sample selection bias in recommender systems. Through joint modeling of CVR and CTR tasks, it performs parameter learning in the entire space. The core innovation lies in introducing CTR as an auxiliary task and optimizing CVR estimation through task multiplication relationship. This design not only solves the sample selection bias in traditional CVR estimation but also provides unbiased CTR and CTCVR estimation.
- **Parameters**:
  - `features` (list): List of features
  - `hidden_units` (list): List of hidden layer units
  - `tower_units` (list): List of task tower layer units
  - `embedding_dim` (int): Feature embedding dimension

### MMoE (Multi-gate Mixture-of-Experts)
- **Introduction**: A multi-task learning model proposed by Google that achieves soft parameter sharing through expert mechanism and task-related gating networks. Each expert network can learn specific feature transformations, while gating networks dynamically allocate expert importance for each task. This design allows the model to flexibly combine expert knowledge based on task requirements, effectively handling task differences.
- **Parameters**:
  - `features` (list): List of features
  - `expert_units` (list): Hidden layer units for expert networks
  - `num_experts` (int): Number of experts
  - `num_tasks` (int): Number of tasks
  - `expert_activation` (str): Activation function for expert networks
  - `gate_activation` (str): Activation function for gate networks

### PLE (Progressive Layered Extraction)
- **Introduction**: An improved version of MMoE that better models task relationships through progressive layered extraction. Introduces the concept of task-specific experts and shared experts, implementing progressive feature extraction through multi-level expert networks. Each layer contains both task-specific experts and shared experts, allowing the model to learn both commonalities and individualities of tasks. This progressive design enhances the model's ability for knowledge extraction and transfer.
- **Parameters**:
  - `features` (list): List of features
  - `expert_units` (list): Units for expert networks
  - `num_experts` (int): Number of experts per layer
  - `num_layers` (int): Number of layers
  - `num_shared_experts` (int): Number of shared experts
  - `task_types` (list): List of task types

