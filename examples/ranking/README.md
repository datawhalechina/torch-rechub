# 排序与多任务示例数据

以下训练命令在 `examples/ranking/` 目录运行。先用仓库随附的样本检查输入和训练流程，需要比较模型效果时再换完整数据。

## Criteo

Criteo 的一行是一条广告曝光记录，一共 40 列。第一列是点击标签（1 表示点击，0 表示未点击），其余是 13 个连续字段和 26 个类别字段。原始文件使用带标签的 `train.txt`。[原始数据](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download) · [完整 CSV](https://aistudio.baidu.com/dataset/detail/389642)

```shell
python run_criteo.py --model_name widedeep
python run_criteo.py --model_name deepfm
```

## Amazon-Electronics

这里使用 2014 年版 Amazon 评论数据。Electronics 部分约有 19 万名用户和 6 万件商品。预处理把评论记录和商品类目合并成 `user_id`、`item_id`、`cate_id`、`time` 四列，DIN 示例据此生成用户的浏览历史和当前要打分的商品。仓库中的样本位于 `data/amazon-electronics/amazon_electronics_sample.csv`。[预处理脚本](./data/amazon-electronics/preprocess_amazon_electronics.py) · [原始数据](http://jmcauley.ucsd.edu/data/amazon/index_2014.html) · [完整 CSV](https://aistudio.baidu.com/dataset/detail/355993)

```shell
python run_amazon_electronics.py
```

## Amazon-Beauty

Beauty 包含约 120 万名用户、24 万件商品和 200 万条评分记录。下载原始数据时选择 Beauty 的 ratings 文件。[原始评分](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) · [处理后的数据](https://aistudio.baidu.com/dataset/detail/391611)

## Amazon-Books

Books 包含约 800 万名用户、200 万种图书和 2,000 万条评分记录。下载原始数据时选择 Books 的 ratings 文件。[原始评分](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) · [处理后的数据](https://aistudio.baidu.com/dataset/detail/391614)

## Ali-CCP

Ali-CCP 收录手机淘宝上的曝光记录，每行有点击和购买两个标签，输入包含 23 个类别字段和 8 个连续字段。完整数据约有 25 万名用户、50 万件商品和 8,000 万条记录。

处理后的类别字段已经编码，连续字段已经归一化。原始数据给出了训练集和测试集，预处理再把原测试集随机平分为验证集和测试集，三份数据的规模约为 2∶1∶1。[仓库预处理脚本](./data/ali-ccp/preprocess_ali_ccp.py) · [AITM 预处理参考](https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py) · [原始数据](https://tianchi.aliyun.com/dataset/408) · [处理后的数据](https://aistudio.baidu.com/datasetdetail/281072)

## Avazu

Avazu 竞赛提供连续 11 天的广告曝光记录。前 10 天约 4,000 万条记录带有点击标签，第 11 天约 460 万条记录只供提交预测。要在本地计算验证和测试指标，可以从有标签的训练数据中按 7∶1∶2 划分三份。[竞赛数据](https://www.kaggle.com/competitions/avazu-ctr-prediction/overview)

常用字段如下。

| 字段名           | 解释                                                             |
| ---------------- | ---------------------------------------------------------------- |
| id               | 每次广告曝光的唯一标识 |
| click            | 广告是否被点击，1表示点击，0表示未点击                           |
| hour             | 曝光时间，格式为 YYMMDDHH                                     |
| C1               | 广告类别1，用于区分广告类别                                      |
| banner_pos       | 广告位置                                                         |
| site_id          | 广告所属站点id                                                   |
| site_domain      | 广告所属站点域名                                                 |
| site_category    | 站点分类                                                         |
| app_id           | 广告所属app id                                                   |
| app_domain       | 广告所属app域名                                                  |
| app_category     | 广告所属app分类                                                  |
| device_id        | 广告所属设备id                                                   |
| device_ip        | 广告所属设备ip                                                   |
| device_model     | 广告所属设备型号                                                 |
| device_type      | 广告所属设备类型                                                 |
| device_conn_type | 广告所属设备连接类型                                             |
| C14-C21          | 匿名化的分类变量                                                 |

## 测试结果

> 下表保留了以往实验记录，分类指标为 AUC，回归指标为 MSE。表中没有逐次记录完整的数据切分与训练配置。使用仓库内置样本验证代码时无需达到这些数值，可复查的对比请参阅 [Benchmark](../../benchmarks/README.md)。

常见排序模型测试结果：

| Model/Dataset | Criteo | Avazu  | Taobao(CTR) |
| ------------- | ------ | ------ | ----------- |
| WideDeep      | 0.8083 | 0.7533 |             |
| DeepFM        | 0.8104 | 0.7579 |             |
| DCN           |        | 0.7533 |             |

序列模型测试结果：

| Model/Dataset | Amazon | Taobao(CTR) |
| ------------- | ------ | ----------- |
| DIN           | 0.8691 |             |

## 多任务学习数据

## AliExpress

AliExpress 有 16 个类别字段和 63 个已经归一化的连续字段，并记录曝光、点击和转化。原数据按地区分成 5 个 CSV，下表中的实验使用美国分区。[原始数据](https://aistudio.baidu.com/datasetdetail/172300) · [处理后的数据](https://aistudio.baidu.com/aistudio/datasetdetail/321095)

## Census-Income

这个例子用同一条人口普查记录预测两个结果。看训练日志时，按以下方式读两列标签。

- `cvr_label` 来自 `income`。原始收入档为 `50000+` 时记为 1。
- `ctr_label` 来自 `marital status`。原始婚姻状态为 `Never married` 时记为 1。
- ESMM 另用 `ctcvr_label = cvr_label * ctr_label` 作为第三列标签。它表示前两个条件同时成立。

训练命令默认读取仓库中的 Census 样本。需要核对字段和编码时可直接查看[预处理脚本](./data/census-income/preprocess_census.py)和[训练脚本](./run_census.py)。[原始数据](https://archive.ics.uci.edu/dataset/117/census+income+kdd) · [处理后的数据](https://aistudio.baidu.com/dataset/detail/390912)

以下命令选择一条运行。

```shell
python run_census.py --model_name SharedBottom --device cpu
python run_census.py --model_name ESMM --device cpu
python run_census.py --model_name MMOE --device cpu
python run_census.py --model_name PLE --device cpu
python run_census.py --model_name AITM --device cpu
```

多任务学习模型测试结果：

| Model\Dataset    | Census-Income(CVR) | Census-Income(CTR) | Taobao(CVR) | Taobao(CTR) | AliExpress-US(CVR) | AliExpress-US(CTR) |
| :--------------- | ------------------ | ------------------ | ----------- | ----------- | ------------------ | ------------------ |
| Shared-Bottom    | 0.9560             | 0.9948             |             |             | 0.8667             | 0.6967             |
| ESMM             | 0.7223             | 0.9906             |             |             |                    |                    |
| MMOE             | 0.9579             | 0.9950             |             |             |                    |                    |
| PLE(num_level=1) | 0.9583             | 0.9951             |             |             |                    |                    |
| PLE(num_level=2) | 0.9593             | 0.9950             |             |             |                    |                    |
| AITM             | 0.9595             | 0.9951             |             |             | 0.8613             | 0.6991             |

> Census-Income 的收入和婚姻状态是同一条人口统计记录的两个属性。这里用标签乘积演示 ESMM 的训练方式，解读表格时要与真实的“先点击、再购买”任务区分开。

在采样后的 AliExpress-US 数据上，历史实验还分别记录了使用 MetaBalance 与 Adam 时的结果。

| Model\Dataset    | CVR(Adam) | CTR(Adam) | CVR(MetaBalance) | CTR(MetaBalance) |
| :--------------- | --------- | --------- | ---------------- | ---------------- |
| Shared-Bottom    | 0.6247    | 0.4868    | 0.7794           | 0.6027           |
| MMOE             | 0.6601    | 0.5856    | 0.6660           | 0.6289           |
| PLE(num_level=1) | 0.7284    | 0.6351    | 0.7511           | 0.6373           |
| AITM             | 0.5970    | 0.4839    | 0.7379           | 0.6093           |

