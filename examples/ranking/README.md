# 使用案例

## Criteo

该数据集是Criteo Labs发布的在线广告数据集。 它包含数百万个展示广告的点击反馈记录，该数据可作为点击率(CTR)预测的基准。 数据集具有40个特征，第一列是标签，其中值1表示已点击广告，而值0表示未点击广告。 其他特征包含13个dense特征和26个sparse特征。

- 原始数据地址：https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download

  使用其中带label的`train.txt`。

- 全量csv数据下载：https://cowtransfer.com/s/3f5e873a254b43

- 使用方法

  ```shell
  python run_criteo.py --model_name widedeep
  python run_criteo.py --model_name deepfm
  ```

  

## Amazon-Electronics

该数据集是 2014 年亚马逊发布的评论数据集（有2014和2018两个版本，注意区分）。该数据集包括评论（评分、文本、帮助投票）、产品元数据（描述、类别信息、价格、品牌和图像特征） 和链接（查看）。 

该数据集对不同种类商品进行了分类，Electronics数据集该类目包含19W用户、6W商品的信息。

- 注意事项
  - 原始数据是json格式，包含两个文件reviews和meta，reviews包含用户交互日志，meta是商品侧特征，因为原始数据含有较多评论信息，数据较大，我们提供了预处理之后的数据，以csv格式保存，并提供下载链接。同时我们选取了前100条数据放在data/sample中。
  - 预处理完的数据仅包含user_id, item_id, cate_id, time四个特征列，而且已经每个特征Lable Encode好了，我们也提供了处理原始数据的脚本`preprocess_amazon.py`。

- 原始数据地址：http://jmcauley.ucsd.edu/data/amazon/index_2014.html  进入之后选择Electronics数据集

- 预处理后的全量数据下载地址：https://cowtransfer.com/s/e911569fbb1043 

- 使用方法

  ```shell
  python run_amazon_electronics.py # run DIN model on sample data
  ```



## Amazon-Beauty

该数据集是 2014 年亚马逊发布的评论数据集（有2014和2018两个版本，注意区分）。该数据集包括评论（评分、文本）、产品元数据（描述、类别信息、价格、品牌和图像特征） 和链接（甚至有封面图）。 

该数据集对不同种类商品进行了分类，Beauty数据集该类目包含120W个用户、24W条商品的信息，共计200W条数据。

- 原始数据地址：http://jmcauley.ucsd.edu/data/amazon/index_2014.html  进入之后选择Beauty中ratings数据集；
- 预处理后的全量数据下载地址：https://cowtransfer.com/s/0765971f36e44d



## Amazon-Books

该数据集是 2014 年亚马逊发布的评论数据集（有2014和2018两个版本，注意区分）。该数据集包括评论（评分、文本）、产品元数据（描述、类别信息、价格、品牌和图像特征） 和链接（甚至有封面图）。 

该数据集对不同种类商品进行了分类，Beauty数据集该类目包含800W个用户、200W条商品的信息，共计2000W条数据。

- 原始数据地址：http://jmcauley.ucsd.edu/data/amazon/index_2014.html  进入之后选择Books中ratings数据集；
- 预处理后的全量数据下载地址：https://cowtransfer.com/s/2c73f60d514049



## WeChat

TBD



## Ali-CCP

该数据集采集自手机淘宝移动客户端的推荐系统日志，一共有23个sparse特征，8个dense特征，包含“点击”、“购买”两个标签。

数据集该类目包含25W个用户、50W条商品的信息，共计8000W条数据。

- 注意事项
  - 原始数据已划分训练集和测试集，预处理过程将原始数据的测试集随机划分一半作为验证集，预处理后的训练集、验证集、测试集比例为2:1:1。
  - 预处理完的数据集的sparse特征已经Lable Encode，dense特征采用归一化处理。数据预处理方式参考[AITM模型预处理脚本](https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py)，我们也提供了处理原始数据的脚本`preprocess_ali_ccp.py`。
- 原始数据地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=408
- 预处理后的全量数据下载地址：https://cowtransfer.com/s/1903cab699fa49



## Avazu

该数据集来源于kaggle赛题:https://www.kaggle.com/competitions/avazu-ctr-prediction/overview
提供了11天的Avazu广告数据来构建和测试模型。
原始数据规模：
train:按时间顺序排列的10天点击数据，约四千万样本。
test:用于测试模型预测的1天广告数据，约四百六十万。
由于Kaggle中的test缺少label，因此推荐使用avazu的划分方法，即把原始的训练集按照7:1:2进行划分。
数据字段极其解释:
| 字段名           | 解释                                                             |
| ---------------- | ---------------------------------------------------------------- |
| id               | 广告点击id，每个点击事件都有一个唯一的id，可以用来做广告点击跟踪 |
| click            | 广告是否被点击，1表示点击，0表示未点击                           |
| hour             | 广告点击时间，格式为YYMMDDHH                                     |
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

> 表格中score值，分类任务均为AUC(越大越好)，回归任务均为MSE(越小越好)

常见排序模型测试结果：

| Model/Dataset | Criteo | Avazu  | Taobao(CTR) |
| ------------- | ------ | ------ | ----------- |
| WideDeep      | 0.8083 | 0.7533 |             |
| DeepFM        | 0.8104 | 0.7579 |             |
| DCN           |        | 0.7533 |             |
| xDeepFM       |        |        |             |

序列模型测试结果：

| Model/Dataset | Amazon | Taobao(CTR) |
| ------------- | ------ | ----------- |
| DIN           | 0.8691 |             |
|               |        |             |
|               |        |             |

# 多任务学习排序

## AliExpress

阿里速卖通数据，一共有16个sparse特征，63个dense特征，包含“曝光”、“点击”、“转化”三个标签，原始数据已对dense特征进行归一化预处理。原始数据一共有5个csv，这里只使用US地区数据进行测试。

- 原始数据地址:https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690&lang=en-us

- 预处理后数据：https://cowtransfer.com/s/7080e52e5f4f4a

## Census-Income

该数据集内容为美国的人口普查收入数据，用于预测收入（50k-，50k+）。预处理之后，一共包含41列，其中7列为dense特征，33列为sparse特征，1列为label。

- 注意事项
  - 在论文MMOE的实验组1与PLE中，均将收入预测作为主任务，婚姻状态预测作为辅助任务。
  - 为了统一对所有多任务模型进行实验，我们按ESMM的设定，将收入预测作为CTR任务，将婚姻状态预测作为CVR任务。
  - 对原始数据的处理参考`preprocess_census.py`
- 原始数据地址：http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
- 预处理之后的数据下载地址：https://cowtransfer.com/s/e8b67418ce044c
- 使用方法

```shell
python run_census.py --model_name SharedBottom
python run_census.py --model_name ESMM
python run_census.py --model_name MMOE
python run_census.py --model_name PLE
python run_census.py --model_name AITM
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

> Note: ESMM中CVR较低正常，因为我们构造了一个虚拟的任务依赖关系，以产生CTCVR label



多任务学习模型在采样后的AliExpress-US上进行MetaBalance和Adam的对比

| Model\Dataset    | CVR(Adam) | CTR(Adam) | CVR(MetaBalance) | CTR(MetaBalance) |
| :--------------- | --------- | --------- | ---------------- | ---------------- |
| Shared-Bottom    | 0.6247    | 0.4868    | 0.7794           | 0.6027           |
| MMOE             | 0.6601    | 0.5856    | 0.6660           | 0.6289           |
| PLE(num_level=1) | 0.7284    | 0.6351    | 0.7511           | 0.6373           |
| AITM             | 0.5970    | 0.4839    | 0.7379           | 0.6093           |

