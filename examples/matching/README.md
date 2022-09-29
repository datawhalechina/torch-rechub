# 召回

## Movielens

使用ml-1m数据集，使用其中原始特征7个user特征`'user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip',"cate_id"`，2个item特征`"movie_id", "cate_id"`，一共9个sparse特征。

- 构造用户观看历史特征``hist_movie_id``，使用`mean`池化该序列embedding
- 使用随机负采样构造负样本 (sample_method=0)，内含随机负采样、word2vec负采样、流行度负采样、Tencent负采样等多种方法
- 将每个用户最后一条观看记录设置为测试集
- 原始数据下载地址：https://grouplens.org/datasets/movielens/1m/
- 处理完整数据csv下载地址：https://cowtransfer.com/s/5a3ab69ebd314e

以下指标使用example中的相同参数在ml-1m上测试得到

| Model\Metrics | Hit@100 | Recall@100 | Precision@100 |
|---------------|---------|------------|---------------|
| DSSM          | 14.74%  | 14.74%     | 0.15%         |
| YoutubeDNN    | 6.27%   | 6.27%      | 0.06%         |
| FacebookDSSM  | 2.85%   | 2.85%      | 0.03%         |
| YoutubeSBC    | 17.83%  | 17.83%     | 0.18%         |



## 双塔模型对比

| 模型         | 学习模式   | 损失函数  | 样本构造                                                     | label                              |
| ------------ | ---------- | --------- | ------------------------------------------------------------ | ---------------------------------- |
| DSSM         | point-wise | BCE       | 全局负采样，一条负样本对应label 0                            | 1或0                               |
| YoutubeDNN   | list-wise  | CE        | 全局负采样，每条正样本对应k条负样本                          | 0（item_list中第一个位置为正样本） |
| YoutubeSBC   | list-wise  | CE        | Batch内随机负采样，每条正样本对应k条负样本，加入采样权重做纠偏处理 | 0（item_list中第一个位置为正样本） |
| FacebookDSSM | pair-wise  | BPR/Hinge | 全局负采样，每条正样本对应1个负样本，需扩充负样本item其他属性特征 | 无label                            |



## YiDian-News
一点资讯-CTR比赛数据集

### 全量数据
* 原始数据为NewsDataset.zip[(下载链接)](https://cowtransfer.com/s/7ee14d7550d749) ，包括下面`数据列表`说明的信息。

* `1.1 EDA&preprocess-train_data` and `1.2 EDA&preprocess-user_info` 是对原始数据train_data.txt和user_info.txt的EDA和预处理，输出user_item.pkl和user.pkl。（PS：pkl的读取速度是csv的好几倍，所以存储为pkl格式）

* `2. merge&transform` 读取上一步的输出，将user和user-item连接，将showPos、refresh分桶，将network转为One-Hot向量，输出all_data.pkl。**此notebook对内存要求较高，建议60G以上。** 最终数据量1亿8千万，38列。[最终输出及脚本文件下载地址](https://cowtransfer.com/s/46f663bc4fce42)

#### 数据列表：
1. 用户信息user_info.txt，“\t”分割，各列字段为：用户id、设备名称、操作系统、所在省、所在市、年龄、性别； 
2. 文章信息doc_info.txt，“\t”分割，各列字段为：文章id、标题、发文时间、图片数量、一级分类、二级分类、关键词；
3. 训练数据train_data.txt，“\t”分割，各列字段为：用户id、文章id、展现时间、网路环境、刷新次数、展现位置、是否点击、消费时长（秒）；

#### 数据项说明：
1. 网络环境：0：未知；1：离线；2：WiFi；3：2g；4：3g；5：4g；
2. 刷新次数：用户打开APP后推荐页的刷新次数，直到退出APP则清零；
3. 训练数据取自用户历史12天的行为日志，测试数据采样自第13天的用户展现日志；

#### 参考资料
[比赛链接](https://tech.yidianzixun.com/competition/#/)

[第一参赛者笔记](https://www.logicjake.xyz/2021/09/20/%E4%B8%80%E7%82%B9%E8%B5%84%E8%AE%AF%E6%8A%80%E6%9C%AF%E7%BC%96%E7%A8%8B%E5%A4%A7%E8%B5%9BCTR%E8%B5%9B%E9%81%93-%E8%B5%9B%E5%90%8E%E6%80%BB%E7%BB%93/)



## Session based recommendation datasets
#### 序列推荐的基准数据集目前包括:
* YOOCHOOSE: [ACM RecSys Challenge 2015](https://recsys.acm.org/recsys15/challenge/) 所使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `train-item-views.csv` 文件。该文件包含四个原始特征：`"session_id", "item_id", "time","category"`。该数据目前可以在 [Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) 网站下载。
* DIGINETICA: [CIKM Cup 2016](https://competitions.codalab.org/competitions/11161) 使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `yoochoose-clicks.dat` 文件。该文件包含五个原始特征：`"sessionId", “userId”, "itemId", "time", "timeframe", "eventdate"`。该数据目前可以在 [google drive](https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw) 下载。

#### 测试结果

|       | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | YOOC.1/64<br> Recall@20 | DIGI.<br> Recall@20 | DIGI.<br> Recall@20 |
|:-----:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-------------------:|:-------------------:|
|  NARM |          0.6746         |          0.2827         |          0.7028         |          0.2909         |        0.5829       |        0.2603       |
| STAMP |          0.6675         |          0.2859         |          0.7079         |          0.3074         |        0.5578       |        0.2303       |

* __Neural Attentive Session-based Recommendation__ ([Li et al., CIKM'17](https://dl.acm.
* __STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation__  ([Liu et al., KDD'18](https://dl.acm.org/doi/10.1145/3219819.3219950))
* 注：以上指标可使用论文中实验章节提到的训练参数测试得到。排序指标 `top_k` 则需要调整成 `20`。