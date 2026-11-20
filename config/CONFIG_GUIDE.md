# 项目配置与本地检查

从仓库根目录运行下面的命令。如果只改了站点页面，可以直接看[文档预览](#文档预览)。

## 提交 Python 代码前

```bash
uv sync
uv run python config/format_code.py
git diff
```

脚本先用 isort 排列导入，再用 YAPF 排版，最后运行 Flake8。它会直接修改 `torch_rechub/`、`examples/`、`tests/` 和 `benchmarks/` 里的 Python 文件。运行后查看 `git diff`，确认本次准备提交哪些变化。[格式化脚本](./format_code.py)

只想看一个文件的格式差异，可以分别运行两个工具。以下以 DeepFM 为例。

```bash
uv run isort --profile black --check-only --diff torch_rechub/models/ranking/deepfm.py
uv run yapf --style=pyproject.toml --diff torch_rechub/models/ranking/deepfm.py
```

改了排序模型，可以先跑排序测试。需要覆盖率报告时，在命令末尾加上 `--cov=torch_rechub --cov-report=html`，生成的首页是 `htmlcov/index.html`。

```bash
uv run pytest -c config/pytest.ini tests/test_e2e_ranking.py
```

想在每次提交前自动检查暂存的文件，可运行一次 `uv run pre-commit install`。要手动检查所有文件，运行 `uv run pre-commit run --all-files`。[提交钩子配置](../.pre-commit-config.yaml)

## 到哪里改配置

| 想调整什么 | 看哪个文件 |
| --- | --- |
| Python 包依赖、可选依赖组、YAPF 和 isort 的工具配置 | [pyproject.toml](../pyproject.toml) |
| 格式化脚本检查哪些目录、给各工具传什么参数 | [format_code.py](./format_code.py) |
| 单独运行 Flake8 时指定的规则 | [.flake8](./.flake8)，用 `--config=config/.flake8` 加载 |
| pytest 的测试目录和选项 | [pytest.ini](./pytest.ini)，用 `-c config/pytest.ini` 加载 |
| 文档侧栏和语言切换 | [VitePress 配置](../docs/.vitepress/config.mts) |

## 文档预览

在仓库根目录安装文档依赖，启动预览，再打开终端打印的本地地址。要确认能否生成静态站点，运行 `npm run docs:build`。

```bash
npm ci
npm run docs:dev
```

## GitHub 上会运行哪些检查

向 `main` 提交 PR，且改动涉及源码、示例、测试等受监控路径时，CI 会检查代码格式、运行测试和安全扫描，并尝试构建 Python 包。Ubuntu 运行核心用例，兼容性任务在指定的 Ubuntu 和 Windows 环境运行代表用例。失败时，在 PR 的 Checks 里打开对应任务，找到第一个报错的步骤。[CI 配置及触发路径](../.github/workflows/ci.yml)

站点文档随 `main` 上的相关改动构建并部署到 GitHub Pages。只改 `docs/` 的 PR 可以先在本地构建预览。[文档部署配置](../.github/workflows/deploy.yml)

## 发布版本

维护者在 GitHub 发布 Release 时填写版本标签和说明。发布任务从标签读取版本号，从说明生成更新日志，更新 API 文档后构建安装包。随后它把发布信息同步回 `main`，触发文档部署，并将安装包发布到 PyPI 和 GitHub Release。[发布步骤](../.github/workflows/ci.yml) · [版本与更新日志脚本](./prepare_release.py)
