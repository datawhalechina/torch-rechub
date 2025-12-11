# 🔥 Torch-RecHub - Lightweight, Efficient & Easy-to-use PyTorch Recommender Framework

> 🚀 **30+ Mainstream Models** | 🎯 **Out-of-the-box** | 📦 **One-click ONNX Export** | 🤖 **Generative RecSys (HSTU/HLLM)**

[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/datawhalechina/torch-rechub?style=for-the-badge)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-orange?style=for-the-badge)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.7%2B-orange?style=for-the-badge)](https://pytorch.org/)
[![annoy Version](https://img.shields.io/badge/annoy-1.17%2B-orange?style=for-the-badge)](https://github.com/spotify/annoy)
[![pandas Version](https://img.shields.io/badge/pandas-1.2%2B-orange?style=for-the-badge)](https://pandas.pydata.org/)
[![numpy Version](https://img.shields.io/badge/numpy-1.19%2B-orange?style=for-the-badge)](https://numpy.org/)
[![scikit-learn Version](https://img.shields.io/badge/scikit_learn-0.23%2B-orange?style=for-the-badge)](https://scikit-learn.org/)
[![torch-rechub Version](https://img.shields.io/badge/torch_rechub-0.0.3%2B-orange?style=for-the-badge)](https://pypi.org/project/torch-rechub/)

English | [简体中文](README.md)

**Online Documentation:** https://datawhalechina.github.io/torch-rechub/ (English) | https://datawhalechina.github.io/torch-rechub/zh/ (简体中文)

**Torch-RecHub** — **Build production-grade recommender systems in 10 lines of code**. 30+ mainstream models out-of-the-box, one-click ONNX deployment, letting you focus on business instead of engineering.

![Torch-RecHub Banner](docs/public/img/banner.png)

## 🎯 Why Torch-RecHub?

| Feature | Torch-RecHub | Other Frameworks |
|---------|-------------|------------------|
| Lines of Code | **10 lines** for train+eval+deploy | 100+ lines |
| Model Coverage | **30+** mainstream models | Limited |
| Generative RecSys | ✅ HSTU/HLLM (Meta 2024) | ❌ |
| ONNX Export | ✅ Built-in support | Manual adaptation |
| Learning Curve | Very Low | Steep |

## ✨ Features

* **Modular Design:** Easy to add new models, datasets, and evaluation metrics.
* **PyTorch-based:** Leverages PyTorch's dynamic graph and GPU acceleration capabilities.
* **Rich Model Library:** Covers **30+** classic and cutting-edge recommendation algorithms (matching, ranking, multi-task, generative).
* **Standardized Pipeline:** Provides unified data loading, training, and evaluation workflows.
* **Easy Configuration:** Adjust experiment settings via config files or command-line arguments.
* **Reproducibility:** Designed to ensure reproducible experimental results.
* **ONNX Export:** Export trained models to ONNX format for production deployment.
* **Cross-engine data processing:** PySpark-based data processing and conversion supported for large-scale pipelines.
* **Experiment visualization & tracking:** Unified integration of WandB, SwanLab, and TensorBoardX.
* **Additional Features:** Negative sampling, multi-task learning, etc.

## 📖 Table of Contents

- [🔥 Torch-RecHub - Lightweight, Efficient \& Easy-to-use PyTorch Recommender Framework](#-torch-rechub---lightweight-efficient--easy-to-use-pytorch-recommender-framework)
  - [🎯 Why Torch-RecHub?](#-why-torch-rechub)
  - [✨ Features](#-features)
  - [📖 Table of Contents](#-table-of-contents)
  - [🔧 Installation](#-installation)
    - [Requirements](#requirements)
    - [Installation Steps](#installation-steps)
  - [🚀 Quick Start](#-quick-start)
  - [📂 Project Structure](#-project-structure)
  - [💡 Supported Models](#-supported-models)
  - [📊 Supported Datasets](#-supported-datasets)
  - [🧪 Examples](#-examples)
    - [Ranking (CTR Prediction)](#ranking-ctr-prediction)
    - [Multi-Task Ranking](#multi-task-ranking)
    - [Matching Model](#matching-model)
  - [👨‍💻‍ Contributors](#-contributors)
  - [🤝 Contributing](#-contributing)
  - [📜 License](#-license)
  - [📚 Citation](#-citation)
  - [📫 Contact](#-contact)
  - [⭐️ Star History](#️-star-history)

## 🔧 Installation

### Requirements

* Python 3.9+
* PyTorch 1.7+ (CUDA-enabled version recommended for GPU acceleration)
* NumPy
* Pandas
* SciPy
* Scikit-learn

### Installation Steps

**Stable Version (Recommended for Users):**
```bash
pip install torch-rechub
```

**Latest Version:**
```bash
# Install uv first (if not already installed)
pip install uv

# Clone and install
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
uv sync
```

## 🚀 Quick Start

Here's a simple example of training a model (e.g., DSSM) on the MovieLens dataset:

```bash
# Clone the repository (if using latest version)
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
uv sync

# Run example
python examples/matching/run_ml_dssm.py

# Or with custom parameters:
python examples/matching/run_ml_dssm.py --model_name dssm --device 'cuda:0' --learning_rate 0.001 --epoch 50 --batch_size 4096 --weight_decay 0.0001 --save_dir 'saved/dssm_ml-100k'
```

After training, model files will be saved in the `saved/dssm_ml-100k` directory (or your configured directory).

## 📂 Project Structure

```
torch-rechub/             # Root directory
├── README.md             # Project documentation
├── pyproject.toml        # Project configuration and dependencies
├── torch_rechub/         # Core library
│   ├── basic/            # Basic components
│   │   ├── activation.py # Activation functions
│   │   ├── features.py   # Feature engineering
│   │   ├── layers.py     # Neural network layers
│   │   ├── loss_func.py  # Loss functions
│   │   └── metric.py     # Evaluation metrics
│   ├── models/           # Recommendation model implementations
│   │   ├── matching/     # Matching models (DSSM/MIND/GRU4Rec etc.)
│   │   ├── ranking/      # Ranking models (WideDeep/DeepFM/DIN etc.)
│   │   └── multi_task/   # Multi-task models (MMoE/ESMM etc.)
│   ├── trainers/         # Training frameworks
│   │   ├── ctr_trainer.py    # CTR prediction trainer
│   │   ├── match_trainer.py  # Matching model trainer
│   │   └── mtl_trainer.py    # Multi-task learning trainer
│   └── utils/            # Utility functions
│       ├── data.py       # Data processing utilities
│       ├── match.py      # Matching utilities
│       ├── mtl.py        # Multi-task utilities
│       └── onnx_export.py # ONNX export utilities
├── examples/             # Example scripts
│   ├── matching/         # Matching task examples
│   ├── ranking/          # Ranking task examples
│   └── generative/       # Generative recommendation examples (HSTU, HLLM, etc.)
├── docs/                 # Documentation (VitePress: multi-language, English & Chinese)
├── tutorials/            # Jupyter tutorials
├── tests/                # Unit tests
├── config/               # Configuration files
└── scripts/              # Utility scripts
```

## 💡 Supported Models

The framework currently supports **30+** mainstream recommendation models:

### Ranking Models - 13

| Model | Paper | Description |
|-------|-------|-------------|
| **DeepFM** | [IJCAI 2017](https://arxiv.org/abs/1703.04247) | FM + Deep joint training |
| **Wide&Deep** | [DLRS 2016](https://arxiv.org/abs/1606.07792) | Memorization + Generalization |
| **DCN** | [KDD 2017](https://arxiv.org/abs/1708.05123) | Explicit feature crossing |
| **DCN-v2** | [WWW 2021](https://arxiv.org/abs/2008.13535) | Enhanced cross network |
| **DIN** | [KDD 2018](https://arxiv.org/abs/1706.06978) | Attention for user interest |
| **DIEN** | [AAAI 2019](https://arxiv.org/abs/1809.03672) | Interest evolution modeling |
| **BST** | [DLP-KDD 2019](https://arxiv.org/abs/1905.06874) | Transformer for sequences |
| **AFM** | [IJCAI 2017](https://arxiv.org/abs/1708.04617) | Attentional FM |
| **AutoInt** | [CIKM 2019](https://arxiv.org/abs/1810.11921) | Auto feature interaction |
| **FiBiNET** | [RecSys 2019](https://arxiv.org/abs/1905.09433) | Feature importance + Bilinear |
| **DeepFFM** | [RecSys 2019](https://arxiv.org/abs/1611.00144) | Field-aware FM |
| **EDCN** | [KDD 2021](https://arxiv.org/abs/2106.03032) | Enhanced DCN |

### Matching Models - 12

| Model | Paper | Description |
|-------|-------|-------------|
| **DSSM** | [CIKM 2013](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) | Classic two-tower model |
| **YoutubeDNN** | [RecSys 2016](https://dl.acm.org/doi/10.1145/2959100.2959190) | YouTube deep retrieval |
| **YoutubeSBC** | [RecSys 2019](https://dl.acm.org/doi/10.1145/3298689.3346997) | Sampling bias correction |
| **MIND** | [CIKM 2019](https://arxiv.org/abs/1904.08030) | Multi-interest dynamic routing |
| **SINE** | [WSDM 2021](https://arxiv.org/abs/2103.06920) | Sparse interest network |
| **GRU4Rec** | [ICLR 2016](https://arxiv.org/abs/1511.06939) | GRU for sequences |
| **SASRec** | [ICDM 2018](https://arxiv.org/abs/1808.09781) | Self-attentive sequential |
| **NARM** | [CIKM 2017](https://arxiv.org/abs/1711.04725) | Neural attentive session |
| **STAMP** | [KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3219895) | Short-term attention memory |
| **ComiRec** | [KDD 2020](https://arxiv.org/abs/2005.09347) | Controllable multi-interest |

### Multi-Task Models - 5

| Model | Paper | Description |
|-------|-------|-------------|
| **ESMM** | [SIGIR 2018](https://arxiv.org/abs/1804.07931) | Entire space multi-task |
| **MMoE** | [KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3220007) | Multi-gate mixture-of-experts |
| **PLE** | [RecSys 2020](https://dl.acm.org/doi/10.1145/3383313.3412236) | Progressive layered extraction |
| **AITM** | [KDD 2021](https://arxiv.org/abs/2105.08489) | Adaptive information transfer |
| **SharedBottom** | - | Classic shared bottom |

### Generative Recommendation - 2

| Model | Paper | Description |
|-------|-------|-------------|
| **HSTU** | [Meta 2024](https://arxiv.org/abs/2402.17152) | Hierarchical Sequential Transduction Units, powering Meta's trillion-parameter RecSys |
| **HLLM** | [2024](https://arxiv.org/abs/2409.12740) | Hierarchical LLM for recommendation, combining LLM semantic understanding |

## 📊 Supported Datasets

The framework provides built-in support or preprocessing scripts for the following common datasets:

* **MovieLens**
* **Amazon**
* **Criteo**
* **Avazu** 
* **Census-Income**
* **BookCrossing**
* **Ali-ccp**
* **Yidian**
* ...

The expected data format is typically an interaction file containing:
- User ID
- Item ID 
- Rating (optional)
- Timestamp (optional)

For specific format requirements, please refer to the example code in the `tutorials` directory.

You can easily integrate your own datasets by ensuring they conform to the framework's data format requirements or by writing custom data loaders.


## 🧪 Examples

All model usage examples can be found in `/examples`

### Ranking (CTR Prediction)

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
ctr_trainer.export_onnx("deepfm.onnx")
```

### Multi-Task Ranking

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

task_types = ["classification", "classification"]
model = MMOE(features, task_types, 8, expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

mtl_trainer = MTLTrainer(model)
mtl_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
mtl_trainer.export_onnx("mmoe.onnx")
```

### Matching Model

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

model = DSSM(user_features, item_features, temperature=0.02,
             user_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu',
             },
             item_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu',
             })

match_trainer = MatchTrainer(model)
match_trainer.fit(train_dl)
match_trainer.export_onnx("dssm.onnx")
# For two-tower models, you can also export user and item towers separately:
# match_trainer.export_onnx("dssm_user.onnx", tower="user")
# match_trainer.export_onnx("dssm_item.onnx", tower="item")
```

## 👨‍💻‍ Contributors

Thanks to all contributors!

![GitHub contributors](https://img.shields.io/github/contributors/datawhalechina/torch-rechub?color=32A9C3&labelColor=1B3C4A&logo=contributorcovenant)

[![contributors](https://contrib.rocks/image?repo=datawhalechina/torch-rechub)](https://github.com/datawhalechina/torch-rechub/graphs/contributors)

## 🤝 Contributing

We welcome contributions in all forms! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

We also welcome bug reports and feature suggestions through [Issues](https://github.com/datawhalechina/torch-rechub/issues).

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📚 Citation

If you use this framework in your research or work, please consider citing:

```bibtex
@misc{torch_rechub,
    title = {Torch-RecHub},
    author = {Datawhale},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/datawhalechina/torch-rechub}},
    note = {A PyTorch-based recommender system framework providing easy-to-use and extensible solutions}
}
```

## 📫 Contact

* **Project Lead:** [1985312383](https://github.com/1985312383)
* [**GitHub Discussions**](https://github.com/datawhalechina/torch-rechub/discussions)



## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/torch-rechub&type=Date)](https://www.star-history.com/#datawhalechina/torch-rechub&Date)

---

*Last updated: [2025-12-11]*
