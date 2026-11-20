---
title: Training retrieval models
description: How retrieval models use loss functions, vector similarity, and temperature scaling.
---

# Training retrieval models

## 1. What does the model compare in a training example?

A single viewing event can produce several kinds of training examples. Suppose a user watched movie A. The `mode` of `MatchTrainer` determines how the trainer uses the scores returned by the model.

### 1.1 Score each user–item pair (mode = 0)

Each row contains a user, a movie, and a label: 1 for a positive example and 0 for a negative. By default, DSSM returns a score between 0 and 1. The trainer uses binary cross-entropy (BCE) to compare that score with the label.

### 1.2 Compare a positive and a negative item (mode = 1)

For the same user, take a movie A they watched and a sampled movie B. FaceBookDSSM returns two scores. The BPR loss depends on their difference: the higher A scores above B, the lower the loss. The two scores do not need separate targets of 1 and 0.

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log\sigma(s_i^+ - s_i^-)
$$

Here $s_i^+$ and $s_i^-$ are the positive and negative scores for user example $i$. [Derivation of BPR](https://www.cnblogs.com/pinard/p/9128682.html)

### 1.3 Select the positive from a candidate set (mode = 2)

For a candidate list `[A, B, C, D]` with A as the positive, the target label is its position, `0`. YoutubeDNN scores each movie. The trainer passes all four raw scores to cross-entropy, which rewards selecting A from these candidates.

With in-batch sampling, DSSM also uses other users' positive items in the batch as candidates. The trainer then uses cross-entropy even with `mode=0`. For an example of how to construct a batch, see the [retrieval tutorial](/tutorials/retrieval).

## 2. Which distance should I use for retrieval?

In DSSM, both towers produce unit-length vectors. Training scores them with a dot product, while an item index may offer distances such as cosine or angular. How do they relate?

### Cosine similarity and the dot product

Cosine similarity divides the dot product of two vectors by the product of their lengths.

$$
\cos(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}
$$

The closer the directions of two vectors, the closer their cosine similarity is to 1. After L2 normalization, the denominator equals 1, so a dot product gives the cosine similarity directly. If you switch models, check whether the exported vectors are also normalized.

### Euclidean distance

For two unit vectors, expanding their squared Euclidean distance gives the following expression.

$$
\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\top b = 2-2\cos(a,b)
$$

Sorting by decreasing cosine similarity gives the same order as sorting by increasing Euclidean distance. The project's Annoy wrapper defaults to `angular`, which also decreases as cosine similarity increases for unit vectors. When merging retrieval results, check which direction each score or distance uses before sorting the movies.

## 3. Why divide scores by a temperature?

Take four candidate movies with A as the positive and cosine scores of `[1, -1, -1, -1]`. A ranks first, but its softmax probability is still below 1. Its cross-entropy loss is approximately

$$
-\log\frac{e^1}{e^1+3e^{-1}} \approx 0.341
$$

Dividing all scores by a temperature of `0.2` gives `[5, -5, -5, -5]`. The loss for the same positive becomes

$$
-\log\frac{e^5}{e^5+3e^{-5}} \approx 0.00013619
$$

The candidate order stays the same. A lower temperature assigns more probability to the highest-scoring candidate. If a negative ranks first, lowering the temperature also amplifies that error. You can check the calculations against [PyTorch's cross-entropy definition](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

By default, DSSM applies a sigmoid to the dot product and trains with BCE, without temperature scaling. With in-batch sampling, `MatchTrainer` uses the model's `temperature` to scale the candidate scores. YoutubeDNN uses this parameter in its default forward pass.
