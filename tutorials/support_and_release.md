# 学习路线、实际覆盖与发布边界

不先搭通用配置框架，沿三个现有入口学习：

1. 排序：examples/ranking/run_criteo.py，先用 WideDeep 理解 Feature → DataGenerator → CTRTrainer → AUC，再替换 DeepFM/DCN。准备该示例所需的 Criteo sample CSV。
2. 召回：examples/matching/run_ml_dssm.py，先理解正负交互 → 双塔 → 用户/物品向量 → TopK。保留该示例原有数据路径和参数；in-batch 与多卡在各自分支审查、合并后再讲。
3. 多任务：examples/ranking/run_ali_ccp_multi_task.py，先学习 SharedBottom 与二维标签，再换 MMOE/PLE。ESMM 明确输出顺序为 [CVR, CTR, CTCVR]，训练 CTR 与 CTCVR，不能把任意两列标签硬塞进去。

## 覆盖矩阵

| 范围 | 本分支固定验证范围 | 尚未覆盖/需要其他分支 |
| --- | --- | --- |
| 排序公开模型 13 个 | 小 CPU 数据：训练、验证、参数确实更新、文件权重重载、预测一致；没有模型级无条件 skip | 全量业务数据、GPU、所有输入组合 |
| DIN | 当前原有的稀疏上下文 + 历史 + target 用法 | 连续上下文存在维度问题；与 padding/in-batch 分支重叠，暂缓模型修改 |
| 多任务公开模型 5 个 | 普通损失模式；正确二维标签；ESMM 三列语义；fit/evaluate/save/load | MetaBalance/GradNorm 与特殊调度组合应在相应修复合并后验收 |
| 召回 | 现有测试明确只覆盖 DSSM/YoutubeDNN/FaceBookDSSM | 剩余模型 fixture 与召回 in-batch 模型改动耦合，未宣称全部通过 |
| 索引与推理产物 | 各自独立 pr/ 分支提供验证与教程 | 这里不把尚未合并的 API 写成主线已具备 |
| 生成式 | 本次仅补齐 sentencepiece extra 依赖 | 论文复现、训练规模与效果不在本次验收范围 |

这不是“所有模型已可生产”的声明。测试验证基本训练契约，不验证论文指标、数据泄漏、跨 GPU 同步或真实线上负载。新增公开排序/多任务模型而未配置 fixture 时测试直接失败，不静默跳过。

本地运行：

```shell
python -m pytest -c config/pytest.ini tests/test_e2e_ranking.py tests/test_e2e_multitask.py
python config/format_code.py
```

Windows 控制台不是 UTF-8 时设置 PYTHONUTF8=1。CI 的 Python 3.10–3.13 兼容矩阵也跑上述训练与重载，而不只 import。它将在远程 CI 执行，本地没有伪装验证这些 Python 版本。当前包仍声明 torch>=1.10；MatchTrainer 的 weights_only 加载与旧版本不兼容，因与召回分支耦合暂缓，不能据此声称已完整验证最低版本。推理产物加载新增 API 明确要求 >=1.13。

## 发布前必须完成的操作

1. 在分支中更新 pyproject.toml 的版本和 CHANGELOG，先通过 PR 的格式、未定义变量检查与测试。
2. 合并后在该 commit 打标签 vX.Y.Z 并创建 Release。
3. 发布工作流只检出这个标签；检查标签版本与 pyproject.toml 一致，不一致即停止；不会临时修改版本、提交或推送 main。
4. 从标签原代码构建 wheel/sdist 并检查元数据，再上传 PyPI 和 Release。PyPI 凭据/可信发布设置仍由仓库维护者配置，本地不实际发布。

本分支取消 F821 全局忽略，同时给 EDCN 的 bridge 状态显式初值，拒绝零/负层数。发布时不再把后来的 main 冒充已打标签版本；文档发布仍由原 main 路径变动触发或人工 workflow_dispatch，不再依赖自动改版提交。
