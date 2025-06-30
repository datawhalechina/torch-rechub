# 配置完整指南

这个文档提供了torch-rechub项目的配置文件组织结构和CI/CD流程的完整指南。

## 📋 目录
- [配置文件组织结构](#配置文件组织结构)
- [CI/CD流程概览](#cicd流程概览)
- [配置文件说明](#配置文件说明)
- [Google Python风格指南](#google-python风格指南)
- [工具链详解](#工具链详解)
- [本地开发设置](#本地开发设置)
- [配置工具](#配置工具)
- [发布流程](#发布流程)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)
- [参考资源](#参考资源)

## 📁 配置文件组织结构

### 最终配置文件布局

为了简化配置管理，我们将配置文件按必要性重新组织：

#### 根目录 (必须的配置文件)
- `pyproject.toml` - **完整配置** - Python工具和包构建配置
- `.pre-commit-config.yaml` - **完整配置** - Git钩子自动化配置

#### config/ 目录 (工具特定配置)
- `config/.style.yapf` - YAPF代码格式化配置（Google风格）
- `config/.flake8` - Flake8代码检查配置  
- `config/pytest.ini` - Pytest测试框架配置
- `config/format_code.py` - 代码格式化脚本
- `config/CONFIG_GUIDE.md` - 本配置完整指南

#### 文档配置
- `mkdocs.yml` - MkDocs文档构建配置

### 🔧 为什么这样组织？

#### 必须在根目录的配置
1. **`pyproject.toml`** - Python生态系统标准，包管理工具默认在根目录查找
2. **`.pre-commit-config.yaml`** - pre-commit工具默认在根目录查找

#### 可以在子目录的配置
- `.style.yapf` 和 `.flake8` - 可以通过命令行参数指定路径
- `pytest.ini` - 可以通过 `-c` 参数指定

### 💡 优势

1. **符合标准** - 必须的配置文件放在根目录，符合Python生态系统惯例
2. **清晰组织** - 可选配置文件按功能分类到config目录
3. **便于维护** - 减少根目录文件数量，保持整洁
4. **工具兼容** - 符合各种工具的默认行为
5. **CI/CD优化** - 配置路径明确，便于自动化

## 🔄 CI/CD流程概览

torch-rechub项目采用现代化的CI/CD流程，确保代码质量和自动化部署。我们的工具链包括：

- **代码格式化**: YAPF (Google风格)
- **导入排序**: isort
- **代码检查**: Flake8
- **类型检查**: MyPy (可选)
- **测试框架**: pytest
- **自动化**: pre-commit + GitHub Actions

### 📋 主要CI流程 (ci.yml)

当代码文件（非文档）变更时触发，包括以下阶段：

#### 1. 代码质量检查 (lint)
- **YAPF**: 代码格式化检查（Google风格）
- **isort**: 导入语句排序检查  
- **Flake8**: 代码风格和语法检查

#### 2. 测试矩阵 (test)
- 支持Python 3.8-3.12版本
- 支持Ubuntu、Windows、macOS操作系统
- 运行pytest测试套件
- 生成测试覆盖率报告

#### 3. 安全检查 (security)
- **Bandit**: 安全漏洞扫描

#### 4. 构建检查 (build)
- 构建Python包
- 验证包的完整性

#### 5. 自动发布 (publish)
- 当创建GitHub Release时自动发布到PyPI
- 使用trusted publishing机制

### 📚 文档流程 (deploy.yml)

当docs目录变更时触发：

#### 1. 文档构建和部署
- 使用MkDocs构建项目文档
- 自动部署到GitHub Pages
- 支持中英文双语文档

### 🚀 触发条件

#### CI流程 (ci.yml) 自动触发
- **Push到main/develop分支** (代码文件变更): 运行完整的CI流程
- **Pull Request** (代码文件变更): 运行完整的CI流程  
- **创建Release**: 额外运行发布流程
- **排除**: docs目录、*.md文件变更

#### 文档流程 (deploy.yml) 自动触发
- **Push到main分支** (docs目录变更): 构建并部署文档
- **Pull Request** (docs目录变更): 构建文档预览

#### 手动触发
可以在GitHub Actions页面手动运行任意workflow

## 📁 配置文件说明

### 根目录配置文件

| 文件 | 作用 | 说明 |
|------|------|------|
| `pyproject.toml` | 项目构建和工具配置 | Python工具配置和包构建 |
| `.pre-commit-config.yaml` | Git钩子配置 | 提交前自动代码检查 |

### config目录配置文件

| 文件 | 作用 | 说明 |
|------|------|------|
| `config/.style.yapf` | YAPF代码格式化配置 | Google风格的详细配置 |
| `config/.flake8` | 代码质量检查配置 | 与Google风格兼容的检查规则 |
| `config/pytest.ini` | 测试框架配置 | 测试发现、覆盖率和标记 |
| `config/format_code.py` | 代码格式化脚本 | 一键格式化所有代码 |
| `config/yapf_config.py` | YAPF配置说明 | 详细的配置选项解释 |

### docs目录配置文件

| 文件 | 作用 | 说明 |
|------|------|------|
| `mkdocs.yml` | MkDocs文档构建配置 | 文档网站构建配置 |

### GitHub Actions配置

| 文件 | 作用 | 说明 |
|------|------|------|
| `.github/workflows/ci.yml` | 主CI流程 | 代码检查、测试、构建、发布 |
| `.github/workflows/deploy.yml` | 文档部署 | MkDocs文档自动部署 |
| `.github/dependabot.yml` | 自动依赖更新配置 | 每周检查依赖更新 |

### 模板文件 (双语)
- `.github/ISSUE_TEMPLATE/bug_report.md` - Bug报告模板
- `.github/ISSUE_TEMPLATE/feature_request.md` - 功能请求模板  
- `.github/pull_request_template.md` - Pull Request模板

## 🚀 使用方法

### 代码格式化
```bash
# 使用脚本格式化（推荐）
python config/format_code.py

# 手动使用工具（自动使用根目录配置）
yapf --style=config/.style.yapf --in-place --recursive torch_rechub/
isort torch_rechub/  # 自动使用pyproject.toml配置
flake8 --config=config/.flake8 torch_rechub/
```

### 测试运行
```bash
# 使用config中的pytest配置
pytest -c config/pytest.ini tests/ -v --cov=torch_rechub
```

### Pre-commit设置
```bash
# 自动使用根目录的配置文件
pip install pre-commit
pre-commit install
```

### 文档构建
```bash
# 使用docs目录中的配置
mkdocs build
```

### CI/CD工具路径

#### isort配置
- **根目录**: `pyproject.toml` - isort会自动使用
- **命令**: `isort torch_rechub/` (无需指定配置文件)

#### YAPF配置
- **config目录**: `config/.style.yapf`
- **命令**: `yapf --style=config/.style.yapf`

#### Flake8配置
- **config目录**: `config/.flake8`
- **命令**: `flake8 --config=config/.flake8`

#### Pytest配置
- **config目录**: `config/pytest.ini`
- **命令**: `pytest -c config/pytest.ini`

## 🎨 Google Python风格指南

### 核心特点

1. **缩进和空格**
   ```python
   # 使用4个空格，不使用tab
   def function_name(param1, param2):
       """函数文档字符串使用三重双引号."""
       if condition:
           return result
   ```

2. **行长度**
   ```python
   # 最大248字符，提高代码可读性，避免过度换行
   very_long_variable_name = some_function_with_long_name(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6)
   ```

3. **字符串格式**
   ```python
   # 优先使用双引号
   message = "Hello, world!"
   
   # 格式化字符串
   greeting = f"Hello, {name}!"
   
   # 多行字符串
   long_text = """这是一个
   多行字符串
   的例子。"""
   ```

4. **导入语句**
   ```python
   # 标准库
   import os
   import sys
   
   # 第三方库
   import numpy as np
   import torch
   
   # 本地模块
   from torch_rechub.basic import layers
   from torch_rechub.models import ranking
   ```

5. **函数和类定义**
   ```python
   class ExampleClass:
       """类的文档字符串."""
   
       def __init__(self, param1: str, param2: int = 0):
           """初始化方法."""
           self.param1 = param1
           self.param2 = param2
   
       def method_name(self, arg1: str) -> str:
           """方法的文档字符串."""
           return f"{self.param1}: {arg1}"
   ```

6. **注释和文档字符串**
   ```python
   def calculate_score(features: torch.Tensor) -> torch.Tensor:
       """计算特征得分.
   
       Args:
           features: 输入特征张量，形状为 (batch_size, feature_dim)
   
       Returns:
           得分张量，形状为 (batch_size, 1)
   
       Raises:
           ValueError: 当features维度不正确时
       """
       # 这是行内注释，前面有两个空格
       score = torch.sum(features, dim=1, keepdim=True)
       return score
   ```

## 🔧 工具链详解

### YAPF (Yet Another Python Formatter)

**作用**: 自动格式化Python代码，确保一致的代码风格。

**特点**:
- 基于Google Python风格指南
- 智能换行和对齐
- 保持代码语义不变
- 可配置的格式化选项

**常用命令**:
```bash
# 格式化单个文件
yapf --style=config/.style.yapf --in-place file.py

# 格式化整个目录
yapf --style=config/.style.yapf --in-place --recursive torch_rechub/

# 检查格式（不修改文件）
yapf --style=config/.style.yapf --diff --recursive torch_rechub/
```

### isort (Import Sorting)

**作用**: 自动排序和组织Python导入语句。

**特点**:
- 按标准库、第三方库、本地模块分组
- 组内按字母排序
- 与YAPF兼容的多行导入格式

**常用命令**:
```bash
# 排序导入语句
isort torch_rechub/

# 检查排序（不修改文件）
isort --check-only --diff torch_rechub/
```

### Flake8 (Code Quality)

**作用**: 检查代码质量、语法错误和风格问题。

**特点**:
- 结合pycodestyle、pyflakes和mccabe
- 可配置的错误忽略
- 与Google风格兼容

**常用命令**:
```bash
# 检查代码质量
flake8 --config=config/.flake8 torch_rechub/

# 检查特定文件
flake8 --config=config/.flake8 torch_rechub/models/ranking/deepfm.py
```

### pytest (Testing Framework)

**作用**: 运行测试、生成覆盖率报告。

**特点**:
- 自动测试发现
- 丰富的断言
- 插件生态系统
- 覆盖率分析

**常用命令**:
```bash
# 运行所有测试
pytest -c config/pytest.ini

# 运行特定测试
pytest -c config/pytest.ini tests/test_basic.py

# 生成覆盖率报告
pytest -c config/pytest.ini --cov=torch_rechub --cov-report=html
```

### pre-commit (Git Hooks)

**作用**: 在git commit之前自动运行代码检查。

**特点**:
- 多工具集成
- 快速失败机制
- 增量检查
- 易于配置

**常用命令**:
```bash
# 安装git钩子
pre-commit install

# 手动运行所有检查
pre-commit run --all-files

# 运行特定检查
pre-commit run yapf
```

## 🛠️ 本地开发设置

### 1. 安装pre-commit
```bash
pip install pre-commit
pre-commit install
```

### 2. 安装开发依赖
```bash
pip install -e .
pip install pytest pytest-cov yapf isort flake8 mypy
```

### 3. 运行代码质量检查
```bash
# 一键格式化所有代码（推荐）
python config/format_code.py

# 检查格式（不修改文件）
python config/format_code.py --check

# 手动运行各工具
yapf --style=config/.style.yapf --in-place --recursive torch_rechub/
isort torch_rechub/  # 自动使用根目录pyproject.toml配置
flake8 --config=config/.flake8 torch_rechub/
mypy torch_rechub/  # 自动使用根目录pyproject.toml配置
```

### 4. 运行测试
```bash
# 使用config目录中的配置
pytest -c config/pytest.ini tests/ -v --cov=torch_rechub
```

### 5. 配置编辑器

#### VS Code
```json
{
    "python.linting.pylintEnabled": false,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--config=config/.flake8"],
    "python.formatting.provider": "yapf",
    "python.formatting.yapfArgs": ["--style=config/.style.yapf"],
    "editor.formatOnSave": true,
    "editor.formatOnType": true
}
```

#### PyCharm
1. 设置 → 工具 → 外部工具 → 添加YAPF
2. 设置 → 代码样式 → Python → 设置为Google风格

## 🔧 配置工具

项目包含便捷的代码格式化脚本：

### 使用方法

```bash
# 基本使用 - 格式化所有代码
python config/format_code.py

# 仅检查格式（不修改文件）
python config/format_code.py --check

# 详细输出
python config/format_code.py --verbose

# 指定特定目录
python config/format_code.py torch_rechub/models/

# 组合选项
python config/format_code.py --check --verbose
```

### 功能特点

- **智能格式化**: 自动检测Python文件并应用Google风格
- **增量处理**: 支持格式化特定目录或文件
- **安全检查**: 格式化前会检查工具是否安装
- **详细报告**: 显示处理文件数和格式化状态
- **灵活配置**: 支持检查模式和详细输出

## 📦 发布流程

### 1. 更新版本号
在`setup.py`中更新版本号

### 2. 创建Release
1. 在GitHub上创建新的Release
2. 使用语义化版本号 (例如: v0.0.4)
3. 填写Release说明

### 3. 自动发布
CI/CD流程会自动：
- 构建包
- 发布到PyPI
- 部署文档

### 4. PyPI发布配置

#### Trusted Publishing设置
1. 在PyPI项目设置中配置trusted publishing
2. 添加GitHub仓库信息：
   - Repository: `datawhalechina/torch-rechub`
   - Workflow: `ci.yml`
   - Environment: `pypi`

## 📊 监控和维护

### GitHub Actions状态
- 在仓库的Actions页面查看CI/CD运行状态
- 失败的workflow会发送邮件通知

### 依赖更新
- Dependabot每周一自动检查依赖更新
- 自动创建PR更新过期依赖

### 测试覆盖率
- 测试覆盖率报告上传到Codecov
- 在PR中显示覆盖率变化

## 🎯 最佳实践

### 日常开发流程

1. **创建分支**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **编写代码**
   - 遵循Google Python风格
   - 添加适当的文档字符串
   - 编写相应的测试

3. **格式化代码**
   ```bash
   python config/format_code.py
   ```

4. **运行测试**
   ```bash
   pytest -c config/pytest.ini
   ```

5. **提交代码**
   ```bash
   git add .
   git commit -m "feat: implement new feature"
   ```

6. **推送并创建PR**
   ```bash
   git push origin feature/new-feature
   ```

### 提交代码前
1. 运行pre-commit hooks
2. 确保所有测试通过
3. 检查代码覆盖率

### 创建PR时
1. 填写完整的PR描述
2. 关联相关的Issue
3. 确保CI检查通过

### 发布新版本时
1. 更新版本号和CHANGELOG
2. 确保所有测试通过
3. 创建详细的Release说明

## 🐛 故障排除

### 常见问题

#### 1. YAPF格式检查失败
```bash
# 原因：代码格式不符合Google风格
# 解决：运行自动格式化
python config/format_code.py
```

#### 2. isort检查失败
```bash
# 原因：导入语句顺序不正确
# 解决：重新排序导入
isort torch_rechub/
```

#### 3. Flake8检查失败
```bash
# 原因：代码质量问题
# 解决：查看具体错误信息并修复
flake8 --config=config/.flake8 torch_rechub/ --show-source
```

#### 4. 测试失败
```bash
# 原因：代码变更破坏了测试
# 解决：修复代码或更新测试
pytest -c config/pytest.ini -v --tb=long
```

#### 5. Pre-commit钩子失败
```bash
# 原因：提交的代码不符合质量标准
# 解决：修复问题后重新提交
pre-commit run --all-files
git add .
git commit -m "fix: resolve code quality issues"
```

#### CI失败
1. 检查错误日志
2. 本地复现问题
3. 修复后重新提交

#### 发布失败
1. 检查PyPI配置
2. 验证trusted publishing设置
3. 确保版本号唯一

#### 文档构建失败
1. 检查MkDocs配置
2. 验证markdown语法
3. 确保所有链接有效

### 配置问题

#### 1. YAPF配置冲突
- 确保`config/.style.yapf`和`pyproject.toml`中的YAPF配置一致
- 检查编辑器是否使用了不同的格式化配置

#### 2. isort与YAPF冲突
- 已在配置中解决，如遇问题请检查`pyproject.toml`中的isort配置

#### 3. Flake8误报
- 可以在`config/.flake8`中添加忽略规则
- 使用`# noqa: E701`注释忽略特定行的检查

### 性能优化

#### 1. 加速pre-commit
```bash
# 仅检查修改的文件
pre-commit run

# 跳过某些钩子
SKIP=mypy pre-commit run
```

#### 2. 加速CI
- 使用缓存机制
- 并行运行测试
- 仅在必要时运行完整测试套件

## 📞 支持

如果遇到配置或CI/CD相关问题，请：
1. 查看GitHub Actions日志
2. 创建Issue并标记为`config`或`ci/cd`标签
3. 联系维护团队

## 📚 参考资源

- [Google Python风格指南](https://google.github.io/styleguide/pyguide.html)
- [YAPF配置文档](https://github.com/google/yapf#configuration)
- [isort配置文档](https://pycqa.github.io/isort/docs/configuration/config_files.html)
- [Flake8配置文档](https://flake8.pycqa.org/en/latest/user/configuration.html)
- [pytest文档](https://docs.pytest.org/)
- [pre-commit文档](https://pre-commit.com/)
- [GitHub Actions文档](https://docs.github.com/en/actions)

---

*最后更新: 2025-06-30*
*维护者: torch-rechub team* 