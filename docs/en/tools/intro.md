---
title: Training tools
description: Track training, inspect model structure and compare recommendation experiments
---

# Training tools

To see whether training has started learning from the data, pass `log_interval=1` to `fit()` in the ranking, retrieval or multi-task examples. The progress bar shows the loss and learning rate for each batch.

For a closer look at your experiment, choose the tool that answers your question.

- **Which epoch should I keep?** Monitor validation AUC or loss. The trainer keeps the best weights and can stop when the metric has gone several epochs without improving. [Early stopping and checkpoints](/tools/callbacks)
- **How do the curves from two runs differ?** Write loss and validation metrics to local TensorBoard logs or send them to your team's W&B or SwanLab project. [Experiment tracking](/tools/tracking)
- **Where does a feature tensor change shape?** Trace a forward pass to see which layers it passes through. View the graph in Jupyter or save it to a file. [Model diagrams](/tools/visualization)
- **Did a different model improve the metrics?** Run both on the same data and settings, then read the comparison table. [Benchmarks](/tools/benchmark)
