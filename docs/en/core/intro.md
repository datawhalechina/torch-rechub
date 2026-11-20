---
title: Core components
description: Follow one movie impression from feature encoding to a training batch and a click score.
---

# Core components

Consider a movie impression. You know which movies the user has already watched, which movie you showed them, and whether they clicked it. To train a ranker, first turn that record into numbers the model can read. The trainer can then compare the predicted click score with the observed label to calculate the loss.

## From one record to a training batch

1. **Encode the features.** Suppose movie `M42` has ID `3` in your vocabulary. If it is the user's only watched movie and the history length is three, `hist_movie_id` is `[3, 0, 0]`. The last two positions are padding. Use the same movie vocabulary for candidates and histories. [Feature definitions](/core/features)
2. **Build a batch.** With `B` records, the candidate IDs have shape `[B]` and the histories have shape `[B, 3]`. `DataGenerator` puts the features in a dictionary and passes them to the trainer with the `B` click labels. [Loading data](/core/data)
3. **Train the ranker.** The model scores each record in the feature dictionary. `CTRTrainer.fit(train_dl, val_dl)` updates the weights on the training set and measures AUC on the validation set after each epoch. [Training and evaluation](/core/evaluation)

## Trainers for other tasks

For retrieval, you need to find candidates in the item catalog. `MatchDataGenerator` prepares separate inputs for users and items. Once the two towers are trained, `MatchTrainer` can compute the embeddings for each side. [Retrieval tutorial](/tutorials/retrieval)

If a record has several labels, you can still batch it with `DataGenerator`. Configure the tasks in `MTLTrainer` in the same order as the label columns so you can read each task's metrics separately. [Multi-task tutorial](/tutorials/pipeline)

HSTU and HLLM use `SequenceDataGenerator` and `SeqTrainer` to predict the next item. RQ-VAE learns semantic IDs for items, and TIGER learns to generate those IDs. Each has its own training script. [Generative models](/models/generative)
