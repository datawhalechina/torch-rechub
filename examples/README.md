# 使用案例

## Criteo

- 使用方法：参考 `run_criteo.py`

## Amazon

该数据集是 2014 年亚马逊发布的评论数据集（有2014和2018两个版本，注意区分）。该数据集包括评论（评分、文本、帮助投票）、产品元数据（描述、类别信息、价格、品牌和图像特征） 和链接（也查看/还购买了图表）。 

该数据集对不同种类商品进行了分类，我们所使用的是Electronics数据集，该类目包含19W用户、6W商品的信息。

- 注意事项：
  - 原始数据是json格式，包含两个文件reviews和meta，reviews包含用户交互日志，meta是商品侧特征，因为原始数据含有较多评论信息，数据较大，我们提供了预处理之后的数据，以csv格式保存，并提供下载链接。同时我们选取了前100条数据放在data/sample中。
  - 预处理完的数据仅包含user_id, item_id, cate_id, time四个特征列，而且已经每个特征Lable Encode好了，我们也提供了处理原始数据的脚本在preprocess_amazon.py文件中。

- 原始数据地址：http://jmcauley.ucsd.edu/data/amazon/index_2014.html  进入之后选择Electronics数据集

- 预处理后的全量数据下载地址：https://cowtransfer.com/s/e911569fbb1043 

## 使用方法

```python
python python3 run_amazon.py # 训练sample文件中的数据，模型采用DIN模型
python python3 run_amazon.py --preprocessed_file_path amazon_electronic_datasets.csv # 训练全量预处理好的数据
```



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

| Model/Dataset | Criteo | Auazu | Amazon |
| ------------- | ------ | ----- | ------ |
| WideDeep      | 0.8083 |       |        |
| DeepFM        | 0.8104 |       |        |
| DIN           |        |       | 0.8579 |

序列模型测试结果：

| Model/Dataset | Amazon | Taobao(CTR) |
| ------------- | ------ | ----------- |
| DIN           | 0.8579 |             |
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

