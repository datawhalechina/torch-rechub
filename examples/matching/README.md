# 召回

## MovieLens

MovieLens DSSM 的用户塔使用 `user_id`、`gender`、`age`、`occupation`、`zip` 和观影历史 `hist_movie_id`。物品塔使用 `movie_id` 和 `cate_id`。

- 历史电影与候选电影共用嵌入表，DSSM 对有效历史位置做平均池化。
- `run_ml_dssm.py` 默认使用 `sample_method=1`，按物品频次的 0.75 次幂采负例。均匀随机采样对应 `sample_method=0`。加 `--in_batch_neg` 则从批内其他正交互取得负例。
- 把每位用户最后一次与电影的交互留给测试，用它检查推荐列表是否命中。

第一次运行可以直接用仓库中的 `ml-1m_sample.csv`。要换完整数据，下载 [GroupLens 的 MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)，把 `users.dat`、`ratings.dat`、`movies.dat` 放在 `examples/matching/data/ml-1m/` 下。从仓库根目录执行下面两步。[预处理脚本](./data/ml-1m/preprocess_ml.py) 会把评分、电影和用户表合成 `data/ml-1m/ml-1m.csv`。备用数据来源见 [issue #201](https://github.com/datawhalechina/torch-rechub/issues/201)。

```bash
cd examples/matching
uv run python data/ml-1m/preprocess_ml.py
uv run --extra annoy python run_ml_dssm.py --dataset_path data/ml-1m/ml-1m.csv --device cpu
```

下表是项目此前用完整 MovieLens 1M 和示例参数得到的记录。每个用户留出一部电影，所以 Hit@100 与 Recall@100 数值相同。

| Model\Metrics | Hit@100 | Recall@100 | Precision@100 |
|---------------|---------|------------|---------------|
| DSSM          | 14.74%  | 14.74%     | 0.15%         |
| YoutubeDNN    | 6.27%   | 6.27%      | 0.06%         |
| FacebookDSSM  | 2.85%   | 2.85%      | 0.03%         |
| YoutubeSBC    | 17.83%  | 17.83%     | 0.18%         |

## 双塔示例的样本和标签

下表按各示例的默认训练方式说明标签的含义。特别留意两种 `0` 的区别，DSSM 中表示负例，YoutubeDNN 中表示正电影在候选列表的第一个位置。

| 模型         | 学习模式   | 损失函数  | 样本构造                                                     | label                              |
| ------------ | ---------- | --------- | ------------------------------------------------------------ | ---------------------------------- |
| DSSM         | point-wise | BCE       | 全局负采样，一条负样本对应label 0                            | 1或0                               |
| YoutubeDNN   | list-wise  | CE        | 全局负采样，每条正样本对应k条负样本                          | 0（item_list中第一个位置为正样本） |
| YoutubeSBC   | list-wise  | CE        | Batch内随机负采样，每条正样本对应k条负样本，加入采样权重做纠偏处理 | 0（item_list中第一个位置为正样本） |
| FacebookDSSM | pair-wise  | BPR       | 全局负采样，每条正样本对应1个负样本，需扩充负样本item其他属性特征 | 按正负对打分，无需分类标签                            |

## YiDian-News

这份比赛数据记录了新闻曝光和点击，可用 `click` 作为点击率预测的标签。

### 全量数据
[下载比赛数据](https://aistudio.baidu.com/dataset/detail/389517)后，处理版的完整数据由 32 个 CSV 分卷组成。

- `yidian_news_processed_part_0001.csv`
- ...
- `yidian_news_processed_part_0032.csv`

调试样例为 `_sample.csv`，包含 1000 行数据。压缩包为 `dataset.zip`。

如需恢复为单个完整 CSV，可在 `YiDian_News_dataset/` 目录运行：

```powershell
python merge_csv_big.py "yidian_news_processed_part_*.csv" yidian_news_processed.csv
```

调试读取和字段映射时，先用 1,000 行的样本即可。

#### 数据列表

处理后的 CSV 共 22 列：

1. 行为字段：`userId`、`itemId`、`showTime`、`network`、`refresh`、`showPos`、`click`、`duration`；
2. 用户字段：`deviceName`、`OS`、`province`、`city`、`age_0_24`、`age_25_29`、`age_30_39`、`age_40plus`、`female`、`male`；
3. 文章字段：`publishTime`、`imageNum`、`cate1`、`cate2`。

#### 数据项说明

1. 网络环境：0 未知；1 离线；2 WiFi；3 2G；4 3G；5 4G；
2. 刷新次数：用户打开 APP 后推荐页的刷新次数，直到退出 APP 则清零；
3. `click` 为二分类标签，0 表示未点击，1 表示点击；
4. `duration` 为消费时长，单位为秒；
5. 原始训练数据取自用户历史 12 天的行为日志，测试数据采样自第 13 天的用户展现日志。

## Million Song Dataset
Million Song Dataset 由 The Echo Nest 和 LabROSA 合作完成。这里用其中的播放记录构造用户与歌曲的交互，每条记录包含匿名用户 ID、歌曲 ID 和播放次数。

- `train_triplets.txt`（约 2.9 GB）记录百万用户对几十万首歌曲的播放情况。调试时可以先读取其中 200 万条。
- `triplets_metadata.db`（约 700 MB）提供发行时间、作者和作者热度等歌曲信息。


[数据集下载](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip)

[数据集说明](http://millionsongdataset.com/tasteprofile/)

## Book-Crossing

Book-Crossing 收录约 27.8 万名匿名用户对 27.1 万本书的 115 万条显式或隐式评分。项目此前使用的合并数据约有 71.9 万条评分。

[数据集下载和说明](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)


## Session based recommendation datasets
会话推荐用同一次访问中的点击顺序预测下一个物品。以下两份数据分别用于电商点击流实验。
* YOOCHOOSE: [ACM RecSys Challenge 2015](https://recsys.acm.org/recsys15/challenge/) 所使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `yoochoose-clicks.dat` 文件。该文件包含四个原始特征：`"session_id", "item_id", "time","category"`。该数据目前可以在 [Kaggle](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) 网站下载。
* DIGINETICA: [CIKM Cup 2016](https://competitions.codalab.org/competitions/11161) 使用的电商网站点击流数据集。构建序列推荐数据只用到该数据集的 `train-item-views.csv` 文件。预处理时主要用到会话 ID、物品 ID 和时间字段。该数据目前可以在 [google drive](https://drive.google.com/drive/folders/0B7XZSACQf0KdXzZFS21DblRxQ3c?resourcekey=0-3k4O5YlwnZf0cNeTZ5Y_Uw) 下载。

按[预处理脚本](./data/session_based/preprocess_session_based.py)选定数据集与切分比例后，用相同候选集比较 Recall@20 等指标。记录测试集切分比例和候选物品范围，便于复查结果。

- [NARM 论文](https://arxiv.org/abs/1711.04725)
- [STAMP 论文](https://dl.acm.org/doi/10.1145/3219819.3219950)
