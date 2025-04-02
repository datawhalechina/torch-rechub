# [Torch-RecHub] - 基于 PyTorch 的轻量推荐系统框架

[![许可证](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE) 
![GitHub Repo stars](https://img.shields.io/github/stars/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/datawhalechina/torch-rechub?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/datawhalechina/torch-rechub?style=for-the-badge)
[![Python 版本](https://img.shields.io/badge/python-3.8%2B-orange?style=for-the-badge)](https://www.python.org/) 
[![PyTorch 版本](https://img.shields.io/badge/pytorch-1.7%2B-orange?style=for-the-badge)](https://pytorch.org/) 
[![annoy 版本](https://img.shields.io/badge/annoy-1.17%2B-orange?style=for-the-badge)](https://pytorch.org/) 
[![pandas 版本](https://img.shields.io/badge/pandas-1.2%2B-orange?style=for-the-badge)](https://pandas.pydata.org/) 
[![numpy 版本](https://img.shields.io/badge/numpy-1.19%2B-orange?style=for-the-badge)](https://numpy.org/) 
[![scikit-learn 版本](https://img.shields.io/badge/scikit_learn-0.23%2B-orange?style=for-the-badge)](https://scikit-learn.org/)
[![torch-rechub 版本](https://img.shields.io/badge/torch_rechub-0.0.3%2B-orange?style=for-the-badge)](https://pypi.org/project/torch-rechub/)

[English](README.md) | 简体中文

**Torch-RecHub** 是一个使用 PyTorch 构建的、灵活且易于扩展的推荐系统框架。它旨在简化推荐算法的研究和应用，提供常见的模型实现、数据处理工具和评估指标。

## ✨ 特性

* **模块化设计:** 易于添加新的模型、数据集和评估指标。
* **基于 PyTorch:** 利用 PyTorch 的动态图和 GPU 加速能力。
* **丰富的模型库:** 包含多种经典和前沿的推荐算法（请在下方列出）。
* **标准化流程:** 提供统一的数据加载、训练和评估流程。
* **易于配置:** 通过配置文件或命令行参数轻松调整实验设置。
* **可复现性:** 旨在确保实验结果的可复现性。
* **其他特性:** 例如，支持负采样、多任务学习等。

## 📖 目录

- [\[Torch-RecHub\] - 基于 PyTorch 的轻量推荐系统框架](#torch-rechub---基于-pytorch-的轻量推荐系统框架)
  - [✨ 特性](#-特性)
  - [📖 目录](#-目录)
  - [🔧 安装](#-安装)
    - [环境要求](#环境要求)
    - [安装步骤](#安装步骤)
  - [🚀 快速开始](#-快速开始)
  - [📂 项目结构](#-项目结构)
  - [💡 支持的模型](#-支持的模型)
  - [📊 支持的数据集](#-支持的数据集)
  - [🧪 示例](#-示例)
    - [精排（CTR预测）](#精排ctr预测)
    - [多任务排序](#多任务排序)
    - [召回模型](#召回模型)
  - [🤝 贡献指南](#-贡献指南)
  - [📜 许可证](#-许可证)
  - [📚 引用](#-引用)
  - [📫 联系方式](#-联系方式)

## 🔧 安装

### 环境要求

* Python 3.8+
* PyTorch 1.7+ (建议使用支持 CUDA 的版本以获得 GPU 加速)
* NumPy
* Pandas
* SciPy
* Scikit-learn

### 安装步骤
- **稳定版**
```bash
pip install torch-rechub
```

-  **最新版（推荐）**
```bash
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
python setup.py install
```


**安装依赖:**
```bash
pip install -r requirements.txt
```



## 🚀 快速开始

以下是一个简单的示例，展示如何使用本框架训练一个模型（例如 MF - Matrix Factorization）在 MovieLens-100k 数据集上：

```bash
# 1. 准备数据 (如果需要预处理)
# python examples\matching\data\ml-1m\preprocess_ml.py

# 2. 训练模型
python run_ml_dssm.py
# 或者直接使用命令行参数覆盖配置
# python run_ml_dssm.py --model_name dssm --device 'cuda:0' --learning_rate 0.001 --epoch 50 --batch_size 4096 --weight_decay 0.0001 --save_dir 'saved/dssm_ml-100k'
```

训练完成后，模型文件将保存在 `saved/dssm_ml-100k` 目录下（或你配置的其他目录）。

## 📂 项目结构

```
torch-rechub/             # 根目录
├── README.md             # 项目文档
├── torch_rechub/         # 核心代码库
│   ├── basic/            # 基础组件
│   ├── models/           # 推荐模型实现
│   │   ├── matching/     # 召回模型（DSSM/MIND/GRU4Rec等）
│   │   └── ranking/      # 排序模型（WideDeep/DeepFM/DIN等）
|   |   └── multi_task/   # 多任务模型（MMoE/ESMM等）
│   ├── trainers/         # 训练器
│   ├── utils/            # 工具函数
├── examples/             # 示例脚本
│   ├── matching/         # 匹配任务示例
│   └── ranking/          # 排序任务示例
├── docs/                 # 文档目录
├── tutorials/            # Jupyter教程
├── setup.py              # 包安装脚本
├── mkdocs.yml            # MkDocs 配置文件
└── requirements.txt      # 项目依赖
```

## 💡 支持的模型

本框架目前支持以下推荐模型：

**通用推荐 (General Recommendation):**

* **[DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf):** Deep Structured Semantic Model
* **[Wide&Deep](https://arxiv.org/abs/1606.07792):** Wide & Deep Learning for Recommender Systems
* **[FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf):** Factorization Machines
* **[DeepFM](https://arxiv.org/abs/1703.04247):** Deep Factorization Machine
* ... 

**序列推荐 (Sequential Recommendation):**

* **[DIN](https://arxiv.org/pdf/1706.06978.pdf):** Deep Interest Network
* **[DIEN](https://arxiv.org/pdf/1809.03672.pdf):** Deep Interest Evolution Network
* **[BST](https://arxiv.org/pdf/1905.06874.pdf):** Behavior Sequence Transformer
* **[GRU4Rec](https://arxiv.org/pdf/1511.06939.pdf):** Gated Recurrent Unit for Recommendation
* **[SASRec](https://arxiv.org/pdf/1808.09781.pdf):** Self-Attentive Sequential Recommendation
* ... 

**多兴趣的推荐 (Multi-Interest Recommendation):**

* **[MIND](https://arxiv.org/pdf/1904.08030.pdf):** Multi-Interest Network with Dynamic Routing
* **[SINE](https://arxiv.org/pdf/2103.06920.pdf):** Self-Interested Network for Recommendation
* ... 

**多任务推荐 (Multi-Task Recommendation):**

* **[ESMM](https://arxiv.org/pdf/1804.07931.pdf):** Entire Space Multi-Task Model
* **[MMoE](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007):** Multi-Task Multi-Interest Network for Recommendation
* **[PLE](https://dl.acm.org/doi/pdf/10.1145/3394486.3403394):** Personalized Learning to Rank
* **[AITM](https://arxiv.org/pdf/2005.02553.pdf):** Adaptive Interest-Task Matching
* ... 

## 📊 支持的数据集

框架内置了对以下常见数据集格式的支持或提供了处理脚本：

* **MovieLens**
* **Amazon**
* **Crite:** 
* **Avaze** 
* **Cesus-Income**
* **BookCrossing**
* **Ali-ccp**
* **Yidian**
* ... 

我们期望的数据格式通常是包含用户 ID、物品 ID、评分（可选）和时间戳（可选）的交互文件。具体格式要求请参考 `tutorials` 目录下的文示例代码。

你可以方便地集成你自己的数据集，只需确保它符合框架要求的数据格式，或编写自定义的数据加载器。


## 🧪 示例

所有模型使用案例参考 `/examples`


### 精排（CTR预测）

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
```

### 多任务排序

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

task_types = ["classification", "classification"] 
model = MMOE(features, task_types, 8, expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

mtl_trainer = MTLTrainer(model)
mtl_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```

### 召回模型

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x y)
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
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！如果你想为这个项目做出贡献，请遵循以下步骤：

1.  **Fork 本仓库:** 点击页面右上角的 "Fork" 按钮。
2.  **进行修改:** 实现你的新功能或修复 Bug。
3.  **提交更改:** `git commit -m "feat: 添加了某某功能"` 或 `fix: 修复了某某问题"` (遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范更佳)。
4.  **推送分支:** `git push origin` 
5. **创建 Pull Request:** 回到 GitHub 原始仓库页面，点击 "New pull request"，选择你的分支与主仓库的 `main` 分支进行比较，并提交 PR。

请确保你的 PR 描述清晰，说明了你所做的更改及其原因。

我们也欢迎通过 [Issues](https://github.com/datawhalechina/torch-rechub/issues) 报告 Bug 或提出功能建议。

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 📚 引用

如果你在研究或工作中使用了本框架，请考虑引用：

```bibtex
@misc{torch_rechub,
    title = {Torch-RecHub},
    author = {Datawhale},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/datawhalechina/torch-rechub}},
    note = {A PyTorch-based recommender system framework providing easy-to-use and extensible solutions}
}
```

## 📫 联系方式

* **项目负责人:** [morningsky](https://github.com/morningsky) 
* [**GitHub Issues**](https://github.com/datawhalechina/torch-rechub/issues)


---

*最后更新: [2025-03-31]*