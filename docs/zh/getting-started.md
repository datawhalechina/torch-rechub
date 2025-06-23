首先，安装Torch-RecHub：

```bash
pip install torch-rechub
```

然后，使用以下代码进行推荐系统模型的训练：

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

### 模型库

#### 模型清单

| 标题          | 标签          | 开发状态 | 开发人员 | 机构          | 会议   | 年份 | URL                                                          | pdf                                                          |
| ------------- | ------------- | -------- | -------- | ------------- | ------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DIN           | 排序,序列建模 | 已完成   | 赖敏材   | 阿里巴巴      | KDD    | 2018 | [https://arxiv.org/abs/1706.06978](https://arxiv.org/abs/1706.06978 "https://arxiv.org/abs/1706.06978") | [1706.06978.pdf](../file/pdf/1706.06978_0xZD_K10S2.pdf "1706.06978.pdf") |
| ESMM          | 排序          | 已完成   | 赖敏材   | 阿里巴巴      | SIGIR  | 2018 | [https://arxiv.org/abs/1804.07931](https://arxiv.org/abs/1804.07931 "https://arxiv.org/abs/1804.07931") | [1804.07931.pdf](../file/pdf/1804.07931_ybf_jOAFRp.pdf "1804.07931.pdf") |
| Youtube-SBC   | 召回          | 已完成   | 赖敏材   | 谷歌          | RecSys | 2019 | [https://research.google/pubs/pub48840/](https://research.google/pubs/pub48840/ "https://research.google/pubs/pub48840/") | [6c8a86c981a62b0126a11896b7f6ae0dae4c3566.pdf](../file/pdf/6c8a86c981a62b0126a11896b7f6ae0dae4c3566_1QYYhqJR8.pdf "6c8a86c981a62b0126a11896b7f6ae0dae4c3566.pdf") |
| DSSM          | 召回          | 已完成   | 赖敏材   | 微软          | CIKM   | 2013 | [https://posenhuang.github.io/papers/cikm2013\_DSSM\_fullversion.pdf](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf "https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf") | [cikm2013\_DSSM\_fullversion.pdf](../file/pdf/cikm2013_DSSM_fullversion_c9ZSdM19XJ.pdf "cikm2013_DSSM_fullversion.pdf") |
| MetaBalance   | 其他          | 已完成   |          | Facebook      | www    | 2022 | [https://arxiv.org/pdf/2203.06801v1.pdf](https://arxiv.org/pdf/2203.06801v1.pdf "https://arxiv.org/pdf/2203.06801v1.pdf") | [2203.06801v1-3.pdf](../file/pdf/2203.06801v1-3_qUTY4TbvSL.pdf "2203.06801v1-3.pdf") |
| Wide & Deep   | 排序          | 已完成   | 赖敏材   | 谷歌          | DLRS   | 2016 | [https://arxiv.org/pdf/1606.07792.pdf](https://arxiv.org/pdf/1606.07792.pdf "https://arxiv.org/pdf/1606.07792.pdf") | [1606.07792.pdf](../file/pdf/1606.07792_l8JrVnuYXA.pdf "1606.07792.pdf") |
| DSSM-Facebook | 召回          | 已完成   | 赖敏材   | Facebook      | KDD    | 2020 | [https://arxiv.org/abs/2006.11632](https://arxiv.org/abs/2006.11632 "https://arxiv.org/abs/2006.11632") | [2006.11632.pdf](../file/pdf/2006.11632_qiN67CrHNs.pdf "2006.11632.pdf") |
| DeepFM        | 排序          | 已完成   | 赖敏材   | 华为          | IJCAI  | 2017 | [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247 "https://arxiv.org/abs/1703.04247") | [1703.04247.pdf](../file/pdf/1703.04247_sFSyE7q3U1.pdf "1703.04247.pdf") |
| SasRec        | 召回          | 进行中   | 王宇宸   | UCSD          | ICDM   | 2018 | https://arxiv.org/abs/1808.09781                             | [1808.09781v1.pdf](../file/pdf/1808.09781v1.pdf "1808.09781v1.pdf") |
| PLE           | 排序          | 已完成   | 赖敏材   | 腾讯          | RecSys | 2020 | [https://dl.acm.org/doi/abs/10.1145/3383313.3412236?casa\_token=4g\_ErWbxWf8AAAAA%3APhbcdBa6b-SXHlpFtKh1Lybjtv48sYV2l1GsPeL43N5Lpih\_GwarAwV5hzxOYUVZoWd8dimltm4czmI](https://dl.acm.org/doi/abs/10.1145/3383313.3412236?casa_token=4g_ErWbxWf8AAAAA%3APhbcdBa6b-SXHlpFtKh1Lybjtv48sYV2l1GsPeL43N5Lpih_GwarAwV5hzxOYUVZoWd8dimltm4czmI "https://dl.acm.org/doi/abs/10.1145/3383313.3412236?casa_token=4g_ErWbxWf8AAAAA%3APhbcdBa6b-SXHlpFtKh1Lybjtv48sYV2l1GsPeL43N5Lpih_GwarAwV5hzxOYUVZoWd8dimltm4czmI") | [2020 (Tencent) (Recsys) \[PLE\] Progressive Layered Extraction (PLE) - A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations.pdf](<../file/pdf/2020 (Tencent) (Recsys) \[PLE] Progressive Layered .pdf> "2020 (Tencent) (Recsys) \[PLE] Progressive Layered Extraction (PLE) - A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations.pdf") |
| AITM          | 排序          | 已完成   | 赖敏材   | 美团          | KDD    | 2021 | [https://arxiv.org/abs/2105.08489](https://arxiv.org/abs/2105.08489 "https://arxiv.org/abs/2105.08489") | [2105.08489-2.pdf](../file/pdf/2105.08489-2_XnVVGxN9GG.pdf "2105.08489-2.pdf") |
| Shared-Bottom | 排序          | 已完成   | 赖敏材   | CMU           | ML     | 1997 | [https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf](https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf "https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf") | [Caruana1997\_Article\_MultitaskLearning.pdf](../file/pdf/Caruana1997_Article_MultitaskLearning_ySprcjzJ6v.pdf "Caruana1997_Article_MultitaskLearning.pdf") |
| DCN           | 排序          | 已完成   | 赖敏材   | 谷歌,斯坦福   | AKDD   | 2017 | [https://arxiv.org/abs/1708.05123](https://arxiv.org/abs/1708.05123 "https://arxiv.org/abs/1708.05123") | [1708.05123.pdf](../file/pdf/1708.05123_f3lKSqxIvw.pdf "1708.05123.pdf") |
| Youtube-DNN   | 召回          | 已完成   | 赖敏材   | 谷歌          | RecSys | 2016 | [https://dl.acm.org/doi/10.1145/2959100.2959190](https://dl.acm.org/doi/10.1145/2959100.2959190 "https://dl.acm.org/doi/10.1145/2959100.2959190") | [2959100.2959190.pdf](../file/pdf/2959100.2959190_jRzTU81Xmq.pdf "2959100.2959190.pdf") |
| MMOE          | 排序          | 已完成   | 赖敏材   | 谷歌          | KDD    | 2018 | [https://dl.acm.org/doi/pdf/10.1145/3219819.3220007](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007 "https://dl.acm.org/doi/pdf/10.1145/3219819.3220007") | [3219819.3220007.pdf](../file/pdf/3219819.3220007_zvaZg_CZ6z.pdf "3219819.3220007.pdf") |
| GRU4Rec       | 召回,序列建模 | 已完成   | 王凯     | 腾讯          | KDD    | 2022 | https://arxiv.org/abs/1511.06939                             |                                                              |
| SASRec        | 召回,序列建模 | 已完成   | 王宇宸   | UC            | ICDM   | 2018 | [https://arxiv.org/pdf/1808.09781.pdf](https://arxiv.org/pdf/1808.09781.pdf "https://arxiv.org/pdf/1808.09781.pdf") | [1808.09781-3.pdf](../file/pdf/1808.09781-3_bmRm284Rxd.pdf "1808.09781-3.pdf") |
| SINE          | 召回          | 已完成   | 康博     | 阿里巴巴      | WSDM   | 2021 | [https://arxiv.org/pdf/2102.09267.pdf](https://arxiv.org/pdf/2102.09267.pdf "https://arxiv.org/pdf/2102.09267.pdf") | [2102.09267.pdf](../file/pdf/2102.09267_cdwBFKPCrj.pdf "2102.09267.pdf") |
| (FAT-)DeepFFM | 排序          | 已完成   | 康博     | 新浪          | arXiv  | 2019 | [https://arxiv.org/pdf/1905.06336.pdf](https://arxiv.org/pdf/1905.06336.pdf "https://arxiv.org/pdf/1905.06336.pdf") | [1905.06336.pdf](../file/pdf/1905.06336_2oH3RMtROA.pdf "1905.06336.pdf") |
| STAMP         | 召回,序列建模 | 已完成   | 康博     | 电子科大      | KDD    | 2018 | [https://dl.acm.org/doi/10.1145/3219819.3219950](https://dl.acm.org/doi/10.1145/3219819.3219950 "https://dl.acm.org/doi/10.1145/3219819.3219950") | [3219819.3219950.pdf](../file/pdf/3219819.3219950_aTMFXHL3JB.pdf "3219819.3219950.pdf") |
| NARM          | 召回,序列建模 | 已完成   | 康博     | 京东,山东大学 | CIKM   | 2017 | [https://arxiv.org/pdf/1711.04725.pdf](https://arxiv.org/pdf/1711.04725.pdf "https://arxiv.org/pdf/1711.04725.pdf") | [1711.00165.pdf](../file/pdf/1711.00165_eosOSOmTfE.pdf "1711.00165.pdf") |
| DCN\_v2       | 排序          | 已完成   | 叶志雄   | 谷歌          | www    | 2021 | [https://arxiv.org/abs/2008.13535](https://arxiv.org/abs/2008.13535 "https://arxiv.org/abs/2008.13535") | [DCN V2 Improved Deep & Cross Network and Practical Lessons.pdf](<../file/pdf/DCN V2 Improved Deep & Cross Network and Practical.pdf> "DCN V2 Improved Deep & Cross Network and Practical Lessons.pdf") |
| EDCN          | 排序          | 已完成   | 叶志雄   | 华为          | KDD    | 2021 | [https://dlp-kdd.github.io/assets/pdf/DLP-KDD\_2021\_paper\_12.pdf](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf "https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf") |                                                              |
| FiBiNet       | 排序          | 已完成   | 叶志雄   | 新浪          | RecSys | 2019 | [https://dl.acm.org/doi/abs/10.1145/3298689.3347043](https://dl.acm.org/doi/abs/10.1145/3298689.3347043 "https://dl.acm.org/doi/abs/10.1145/3298689.3347043") |                                                              |
| DIEN          | 排序,序列建模 | 已完成   | 范涛     | 阿里巴巴      | AAAI   | 2019 | [https://ojs.aaai.org/index.php/AAAI/article/view/4545](https://ojs.aaai.org/index.php/AAAI/article/view/4545) | [4545-Article Text-7584-1-10-20190706.pdf](../file/pdf/4545-Article%20Text-7584-1-10-20190706.pdf "pdf") |
| BST           | 排序,序列建模 | 已完成   | 范涛     | 阿里巴巴      | arXiv  | 2019 | [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874v1) | [pdf](https://arxiv.org/pdf/1905.06874v1 "pdf")              |


[参考资料](参考资料/参考资料.md "参考资料")
