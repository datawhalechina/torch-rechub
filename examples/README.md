# 使用案例

## Criteo

- 使用方法：参考 `run_criteo.py`

## Amazon

## Census-Income

该数据集内容为美国的人口普查收入数据，用于预测收入（50k-，50k+）。预处理之后，一共包含41列，其中7列为dense特征，33列为sparse特征，1列为label。

- 注意事项
  - 在论文MMOE的实验组1与PLE中，均将收入预测作为主任务，婚姻状态预测作为辅助任务。
  - 为了统一对所有多任务模型进行实验，我们按ESMM的设定，将收入预测作为CTR任务，将婚姻状态预测作为CVR任务。

- 原始数据地址：http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)

- 预处理之后的数据下载地址：https://cowtransfer.com/s/e8b67418ce044c (处理脚本可以参考preprocess.py)

- 使用方法：参考 `run_census.py`

## Taobao

TBD

## 测试结果

> 表格中score值，分类任务均为AUC，回归任务均为MSE

常见排序模型测试结果：

| Model/Dataset | Criteo | Auazu |
| ------------- | ------ | ----- |
| WideDeep      | 0.8083 |       |
| DeepFM        | 0.8104 |       |
|               |        |       |

序列模型测试结果：

| Model/Dataset | Amazon | Taobao(CTR) |
| ------------- | ------ | ----------- |
| DIN           | 0.8403 |             |
|               |        |             |
|               |        |             |

多任务学习模型测试结果：

| Model/Dataset    | Census-Income(CVR) | Census-Income(CTR) | Taobao(CVR) | Taobao(CTR) |
| ---------------- | ------------------ | ------------------ | ----------- | ----------- |
| Shared-Bottom    | 0.9560             | 0.9948             |             |             |
| ESMM             | 0.7223             | 0.9906             |             |             |
| MMOE             | 0.9579             | 0.9950             |             |             |
| PLE(num_level=1) | 0.9583             | 0.9951             |             |             |
| PLE(num_level=2) | 0.9593             | 0.9950             |             |             |

> Note: ESMM中CVR较低正常，因为我们构造了一个虚拟的任务依赖关系，以产生CTCVR label

