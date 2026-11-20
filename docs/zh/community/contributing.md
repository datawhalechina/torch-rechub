---
title: 贡献指南
description: Torch-RecHub 项目贡献指南，包括 Bug 报告、功能请求和 Pull Request 流程
---

# 贡献指南

发现示例跑不通、教程少了一步，或想给项目补一个模型，都可以提 Issue 或 PR。报告问题时，请写明运行的示例、Python 和 PyTorch 版本，附上触发问题的命令和报错。这样别人可以从同一步开始排查。[项目 Issues](https://github.com/datawhalechina/torch-rechub/issues)

## 准备本地环境

先在 GitHub 上 Fork 仓库，把下面的 `YOUR_USERNAME` 换成自己的账号，再克隆代码、创建分支。

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

git switch -c pr/update-guide
```

改文档和改 Python 代码用到的工具不同。选这次改动对应的一节即可。

## 修改文档

中文页面放在 `docs/zh/`，英文对应页面在 `docs/en/`。如果改动影响操作步骤或参数，也要同步另一种语言。在仓库根目录运行：

```bash
npm ci
npm run docs:build
```

编辑时可用 `npm run docs:dev` 边写边看。提交前打开改过的页面，点一下相关链接。如果你改的是示例，按新的步骤跑一遍，确认输出与文中描述一致。

## 修改 Python 代码

在仓库根目录运行 `uv sync` 安装开发依赖。改动涉及可选功能时，一起安装对应的依赖，例如用 `uv sync --extra onnx` 准备 ONNX 导出环境。

新增模型可以参考同一任务下的现有实现，在 `examples/` 放一个使用示例，给出需要的特征、启动命令和运行结果。修复已有模型时，先用能复现问题的小样本验证，再运行相关测试。

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_e2e_ranking.py

# 运行并生成覆盖率报告
uv run pytest --cov=torch_rechub
```

多数测试直接在本机运行。要检查 Milvus 集成，先启动本地 Milvus 服务，再在 PowerShell 中运行：

```powershell
# PowerShell
$env:RUN_MILVUS_TESTS = "1"
uv run --extra milvus pytest tests/test_serving.py -m integration
```

提交 Python 改动前，运行 `uv run python config/format_code.py` 格式化代码，再检查本次差异。

## 提交 Pull Request

1. 用一句话写清改动，例如 `docs: 补充排序示例的数据路径` 或 `fix: 修正序列特征的掩码`。
2. 用 `git push -u origin pr/update-guide` 推送到自己的 Fork。如果分支名不同，换成自己的分支名。
3. 在 GitHub 上向原仓库的 `main` 分支创建 PR。按模板写明原先遇到什么问题、改了哪里、用什么命令验证。有相关 Issue 时附上编号。

参与讨论时请遵守项目的[行为准则](https://github.com/datawhalechina/torch-rechub/blob/main/CODE_OF_CONDUCT.md)。
