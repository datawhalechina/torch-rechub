---
title: Matching Models Training Guide
description: Comprehensive guide on loss functions, similarity metrics, and temperature scaling for matching models
---

## I. Understanding Different Loss Functions — 3 Training Methods

In recall tasks, there are generally three training methods: point-wise, pair-wise, and list-wise. In RecHub, we use the ***mode*** parameter to specify the training method, with each method corresponding to a different loss function.

#### 1.1 Point-wise (mode = 0)

> 🥰**Core Idea: Treat recall as binary classification.**

For a recall model, the input is a tuple \<User, Item>, and the output is $P(User, Item)$, representing the user's interest level in the item.

Training objective: For positive samples, the output should be as close to 1 as possible; for negative samples, as close to 0 as possible.

The most commonly used loss function is BCELoss (Binary Cross Entropy Loss).

#### 1.2 Pair-wise (mode = 1)

> 😝**Core Idea: A user's interest in positive samples should be higher than in negative samples.**

For a recall model, the input is a triple \<User, ItemPositive, ItemNegative\>, outputting interest scores $P(User, ItemPositive)$ and $P(User, ItemNegative)$, representing the user's interest scores for positive and negative item samples.

Training objective: The interest score for positive samples should be higher than that for negative samples.

The framework uses BPRLoss (Bayes Personalized Ranking Loss). Here's the loss formula (for more details, see [here](https://www.cnblogs.com/pinard/p/9128682.html "here") - note that there are slight differences between the linked content and the formula below, but the core idea remains the same):

$$
Loss=\frac{1}{N}\sum^N\ _{i=1}-log(sigmoid(pos\_score - neg\_score))
$$

***

#### 1.3 List-wise (mode = 2)

> 😇**Core Idea: A user's interest in positive samples should be higher than in negative samples.**

Wait, isn't this the same as Pair-wise?

Yes! The core idea of List-wise training is the same as Pair-wise, but the implementation differs.

For a recall model, the input is an N+2 tuple $<User, ItemPositive, ItemNeg\_1, ... , ItemNeg\_N>$, outputting interest scores for 1 positive sample and N negative samples.

Training objective: The interest score for the positive sample should be higher than all negative samples.

The framework uses $torch.nn.CrossEntropyLoss$, applying Softmax to the outputs.

> PS: This List-wise approach can be easily confused with List-wise in Ranking. Although they share the same name, List-wise in ranking considers the order relationship between samples. For example, ranking uses order-sensitive metrics like MAP and NDCG for evaluation, while List-wise in Matching doesn't consider order.

## II. How Far Apart Are Two Vectors? — 3 Similarity Metrics

> 🤔Given a user vector and an item vector, how do we measure their similarity?

Let's first define user vector $user \in \mathcal R^D$ and item vector $item\in \mathcal R^D$, where D represents their dimension.

### 2.1 Cosine

From middle school math:

$$
cos(a,b)=\frac{<a,b>}{|a|*|b|}
$$

This represents the angle between two vectors, outputting a real number between \[-1, 1]. We can use this as a similarity measure: the smaller the angle between vectors, the more similar they are.

In all two-tower models in RecHub, cosine similarity is used during the training phase.

### 2.2 Dot Product

This is the inner product of vectors, denoted as $<a,b>$ for vectors a and b.

A simple insight: **If we L2 normalize vectors a and b, i.e., $\tilde{a}=\frac{a}{|a|}\ ,\tilde{b}=\frac{b}{|b|}$, then computing their dot product is equivalent to $cos(a,b)$**. (This is straightforward, so we'll skip the proof)

In fact, this is exactly how all two-tower models in RecHub work: first computing User Embedding and Item Embedding, then applying L2 Norm to each, and finally computing their dot product to get cosine similarity. This approach improves model validation and inference speed.

### 2.3 Euclidean Distance

Euclidean distance is what we commonly understand as "distance" in everyday life.

> 🙋**For L2 normalized vectors a and b, maximizing their cosine similarity is equivalent to minimizing their Euclidean distance**

Why? See the formula below:

$$
\begin{align*}
  EuclidianDistance(a,b)^2 &= \sum_{i=1}^N(a_i-b_i)^2 \\
    &= \sum_{i=1}^Na_i^2+\sum_{i=1}^Nb_i^2-\sum_{i=1}^N2*a_i*b_i\\
    &= 2-2*\sum_{i=1}^Na_i*b_i \\
    &= 2*(1-cos(a,b))
\end{align*}
$$

Two points to explain:

1. From second line to third line, $\sum\ _{i=1}^N a\_i^2=1$. Why? Because a is L2 normalized. Same for b.
2. From third line to fourth line, $\sum_{i=1}^Na_i*b_i$ is the dot product of vectors a and b; since they're L2 normalized, this equals cos.

In RecHub, we use Annoy's Euclidean distance during the validation phase.

> 🙋**Summary: For L2 normalized vectors, maximizing dot product is equivalent to maximizing cosine similarity is equivalent to minimizing Euclidean distance**

## III. How Hot is the Temperature?

> Before proceeding, please make sure you understand the operations in [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (LogSoftmax + NLLLoss). This is crucial for understanding the source code.

Consider a scenario: Using List-wise training with 1 positive sample and 3 negative samples, with cosine similarity as the training metric.

Suppose our model perfectly predicts a training sample, outputting logits (1, -1, -1, -1). Theoretically, the Loss should be 0, or at least very small. However, with CrossEntropyLoss, we get:

$$
-log(exp(1)/(exp(1)+exp(-1)*3))=0.341
$$

But if we divide the logits by a temperature coefficient $temperature=0.2$, making them (5, -5, -5, -5), after CrossEntropyLoss, we get:

$$
-log(exp(5)/(exp(5)+exp(-5)*3))=0.016
$$

This gives us a negligibly small Loss.

In other words, **dividing logits by a temperature expands the upper and lower bounds of each element in the logits, bringing them back into the sensitive range of softmax operations**.

In practice, L2 Norm is commonly used together with temperature scaling.
