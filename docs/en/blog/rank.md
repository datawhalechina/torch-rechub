---
title: Training ranking models
description: How to interpret AUC and LogLoss when evaluating a ranking model.
---

# Training ranking models

Showing a movie to a user creates an impression. Suppose a user clicks one movie and leaves a second movie unclicked. If DeepFM predicts the click probability of each impression, how do we judge its predictions?

- **AUC measures order.** Scores of 0.6 for the clicked movie and 0.4 for the unclicked movie place them in the correct order. Changing the scores to 0.9 and 0.1 preserves the order, so this pair contributes the same amount to AUC.
- **LogLoss also measures the predicted probabilities.** The two sets of predictions have average LogLoss values of about 0.511 and 0.105. The second assigns a higher probability to the observed click and incurs a smaller loss. A confident incorrect prediction incurs a larger loss.

`CTRTrainer` saves the weights when validation AUC strictly improves. In-batch sampling changes the proportion of positive and negative pairs used in training. Evaluation still uses the labels from actual impressions. Follow the [quick start](/guide/quick_start) to obtain your first AUC, then use the [ranking tutorial](/tutorials/ctr) to try another model.
