---
title: Model catalog
description: Choose a Torch-RecHub model for your recommendation task
---

# Model catalog

Look at a single example from your logs. What is known at prediction time, and what are you trying to predict? For a movie recommender, the choice might look like this.

| Question | Data you need | Where to start |
| --- | --- | --- |
| Which few hundred movies should enter the candidate set? | Movies a user has watched and a catalog of candidate movies | [Retrieval models](/models/matching). DSSM computes separate vectors for users and movies and retrieves candidates by similarity. |
| Which candidate movie should appear first? | User, movie, context and click label for each impression | [Ranking models](/models/ranking). DeepFM gives each impression a score that can be used to order the candidates. |
| Will the user click? If they do, will they pay? | Click and conversion labels for the same impression | [Multi-task models](/models/mtl). SharedBottom produces one prediction per target. ESMM also models the relationship between a click and a subsequent conversion. |
| Which movie is the user likely to watch next? | Each user’s viewing history in chronological order | [Generative recommendation models](/models/generative). HSTU scores movies from the history. TIGER generates the semantic ID of the next movie one token at a time. |

## Choosing a model

DeepFM is a useful first ranking model. Once you have chronological user histories, try DIN to see how the candidate movie affects the weight of each past interaction. For retrieval, start with DSSM to go from training to vector search. Try MIND if you want a user’s interests in, say, thrillers and animation to retrieve different candidates.

For multi-task learning, check which labels are available for each impression. A click label alone is enough for single-task ranking. With conversion labels as well, you can compare the AUC of both tasks on the same validation set. For next-item prediction, interaction sequences suffice for HSTU. HLLM additionally needs movie text embeddings, while TIGER needs a semantic ID for each movie.

## Usage examples

Ranking and retrieval use similar training calls. The main difference is how you prepare a batch. These snippets are independent. The [ranking quick start](/guide/quick_start) and [retrieval tutorial](/tutorials/retrieval) define their features and data loaders.

```python
# Ranking example
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

# Create the model
model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

# Create the trainer
trainer = CTRTrainer(model, optimizer_params={"lr": 0.001}, device="cpu")

# Train the model
trainer.fit(train_dataloader, val_dataloader)

# Retrieval example
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

# Create the model
model = DSSM(user_features=user_features, item_features=item_features, temperature=1.0,
             user_params={"dims": [256, 128, 64]}, item_params={"dims": [256, 128, 64]})

# Create the trainer
trainer = MatchTrainer(model, mode=0, device="cpu")

# Train the model
trainer.fit(train_dataloader)
```

## Model papers

* **Ranking** [DeepFM](https://arxiv.org/abs/1703.04247), [Wide&Deep](https://arxiv.org/abs/1606.07792), [DCN](https://arxiv.org/abs/1708.05123), [DCNv2](https://arxiv.org/abs/2008.13535), [DIN](https://arxiv.org/abs/1706.06978), [DIEN](https://arxiv.org/abs/1809.03672), [BST](https://arxiv.org/abs/1905.06874), [AFM](https://arxiv.org/abs/1708.04617), [AutoInt](https://arxiv.org/abs/1810.11921), [FiBiNET](https://doi.org/10.1145/3298689.3347043), [FAT-DeepFFM](https://arxiv.org/abs/1905.06336), [EDCN](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf).
* **Retrieval** [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf), [YouTubeDNN](https://dl.acm.org/doi/10.1145/2959100.2959190), [YouTubeSBC](https://doi.org/10.1145/3298689.3346996), [MIND](https://arxiv.org/abs/1904.08030), [SINE](https://arxiv.org/abs/2102.09267), [GRU4Rec](https://arxiv.org/abs/1511.06939), [SASRec](https://arxiv.org/abs/1808.09781), [NARM](https://arxiv.org/abs/1711.04725), [STAMP](https://doi.org/10.1145/3219819.3219950), [ComiRec](https://arxiv.org/abs/2005.09347).
* **Multi-task learning** [ESMM](https://arxiv.org/abs/1804.07931), [MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007), [PLE](https://dl.acm.org/doi/10.1145/3383313.3412236), [AITM](https://arxiv.org/abs/2105.08489).
* **Sequential and generative recommendation** [HSTU](https://arxiv.org/abs/2402.17152), [HLLM](https://arxiv.org/abs/2409.12740), [TIGER](https://arxiv.org/abs/2305.05065).

DeepFFM is the baseline described in the FAT-DeepFFM paper.

## Contributing a model

To contribute a model, follow the coding and documentation conventions in the [contribution guide](/community/contributing).
