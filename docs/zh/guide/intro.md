---
title: 项目介绍
description: Torch-RecHub 项目架构、特性和设计原理概览
---

# 项目介绍

## 项目概述

**Torch-RecHub** 是一个使用 PyTorch 构建的、灵活且易于扩展的推荐系统框架。它旨在简化推荐算法的研究和应用，提供常见的模型实现、数据处理工具和评估指标。

![Torch-RecHub 横幅](/img/banner.png "Torch-RecHub 横幅")

## 特性

* **模块化设计:** 易于添加新的模型、数据集和评估指标。
* **基于 PyTorch:** 利用 PyTorch 的动态图和 GPU 加速能力。
* **丰富的模型库:** 包含多种经典和前沿的推荐算法。
* **标准化流程:** 提供统一的数据加载、训练和评估流程。
* **易于配置:** 通过配置文件或命令行参数轻松调整实验设置。
* **可复现性:** 旨在确保实验结果的可复现性。
* **易扩展:** 模型训练与模型定义解耦，无basemodel概念。
* **原生函数:** 尽可能使用pytorch原生的类与函数，不做过多定制。
* **模型代码精简:** 在符合论文思想的基础上方便新手学习
* **其他特性:** 例如，支持负采样、多任务学习等。

## 整体架构

![架构设计图](/img/project_framework.jpg "架构设计图")

## 核心组件

Torch-RecHub 采用模块化设计，将推荐系统的核心功能划分为多个组件，包括：

### 1. 特征处理

处理不同类型的特征，包括数值型特征、类别型特征和序列特征。

详情请参考 [特征定义](/zh/core/features) 页面。

### 2. 数据流水线

负责数据加载、预处理和生成数据加载器，支持排序模型和召回模型的数据处理。

详情请参考 [数据流水线](/zh/core/data) 页面。

### 3. 模型库

实现各种推荐模型，包括排序模型、召回模型、多任务模型和生成式推荐模型。

详情请参考 [模型库](/zh/models/intro) 页面。

### 4. 训练与评估

提供统一的训练接口，支持模型训练、评估、预测和ONNX导出功能。

详情请参考 [训练与评估](/zh/core/evaluation) 页面。

### 5. 研发工具

提供各种工具函数，如ONNX导出、模型可视化、回调函数和损失函数等。

详情请参考 [研发工具](/zh/tools/intro) 页面。

## 支持的模型

### 排序模型 (Ranking Models) - 13个

* DeepFM、Wide&Deep、DCN、DCN-v2、DIN、DIEN、BST、AFM、AutoInt、FiBiNET、DeepFFM、EDCN

### 召回模型 (Matching Models) - 12个

* DSSM、YoutubeDNN、YoutubeSBC、MIND、SINE、GRU4Rec、SASRec、NARM、STAMP、ComiRec、FacebookDSSM

### 多任务模型 (Multi-Task Models) - 5个

* ESMM、MMoE、PLE、AITM、SharedBottom

### 生成式推荐 (Generative Recommendation) - 2个

* HSTU、HLLM

## 快速开始

要开始使用 Torch-RecHub，请参考 [快速入门](/zh/guide/quick_start) 页面，了解如何安装框架并运行第一个示例。

## 生产部署

Torch-RecHub 支持将训练好的模型导出为 ONNX 格式，便于部署到生产环境。详情请参考 [生产部署](/zh/serving/intro) 页面。

## 社区贡献

我们欢迎各种形式的贡献！请查看 [贡献指南](/zh/community/contributing) 了解详细的贡献流程。

## 常见问题

如果您在使用过程中遇到问题，请查看 [常见问题](/zh/community/faq) 页面，或在 GitHub 上提交 Issue。