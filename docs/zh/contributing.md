# 贡献指南

非常感谢您对 Torch-RecHub 项目的兴趣并考虑为其做出贡献！您的帮助对项目的发展至关重要。本指南将详细介绍如何参与贡献。

## 如何贡献

我们鼓励各种形式的贡献，包括但不限于：

*   报告 Bug
*   提交功能请求
*   编写或改进文档
*   提交代码修复或新功能（通过 Pull Request）

## 寻找贡献点

您可以从以下几个方面入手：

1.  **查看 Issues**：浏览 [项目 Issues 列表](https://github.com/datawhalechina/torch-rechub/issues)，寻找标记为 `help wanted` 的问题。
2.  **改进现有功能**：如果您在使用中发现可以优化的地方，欢迎提出建议或直接提交改进。
3.  **实现新功能**：如果您有新的想法，建议先创建一个 Issue 进行讨论，以确保其符合项目方向。

## 贡献流程（代码和文档）

我们使用标准的 GitHub Fork & Pull Request 流程来接受代码和文档的贡献。(下列操作您也可以在Github网页上进行)

1.  **Fork 仓库**
    访问 Torch-RecHub 的 [GitHub 仓库页面](https://github.com/datawhalechina/torch-rechub) ，点击右上角的 "Fork" 按钮，将项目复制到您自己的 GitHub 账户下。

2.  **Clone 您的 Fork**
    将您 Fork 的仓库克隆到本地：
    ```bash
    git clone https://github.com/YOUR_USERNAME/torch-rechub.git
    cd torch-rechub
    ```
    请将 `YOUR_USERNAME` 替换为您的 GitHub 用户名。

3.  **设置上游仓库 (Optional but Recommended)**
    添加原始项目仓库作为上游远程仓库，方便同步更新：
    ```bash
    git remote add upstream https://github.com/datawhalechina/torch-rechub.git
    ```

5.  **进行修改**
    在您的新仓库上进行代码编写、文档修改或其他改进。

6.  **确保代码质量**
    *   **代码风格**：请遵循项目现有的代码风格。
    *   **文档**：对于用户可见的更改（如新功能、API 变动），请更新相关文档（README、`docs/` 目录下的文件等）。

7.  **提交更改**
    使用清晰、有意义的 Commit Message 提交您的更改。我们推荐遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
    ```bash
    git add .
    git commit -m "feat: 添加 XXX 模型支持"
    # 或者
    git commit -m "fix: 修复 README 中的拼写错误"
    # 或者
    git commit -m "docs: 更新贡献指南"
    ```

8.  **推送分支**
    将您的本地代码推送到您 Fork 的 GitHub 仓库：
    ```bash
    git push origin main
    ```

9.  **创建 Pull Request (PR)**
    返回您在 GitHub 上的 Fork 仓库页面，您会看到一个提示，建议您基于新推送的分支创建一个 Pull Request。点击该提示或手动导航到 "Pull requests" 标签页，点击 "New pull request"。
    *   **选择分支**：确保基础仓库 (Base repository) 是 `datawhale/torch-rechub` 的 `main` 分支，头部仓库 (Head repository) 是您 Fork 的仓库以及您刚刚推送的代码。
    *   **填写 PR 信息**：
        *   **标题**：简明扼要地描述 PR 的目的，通常可以基于 Commit Message。
        *   **描述**：详细说明您所做的更改、解决的问题（可以链接相关的 Issue，例如 `Closes #123`）、以及任何需要评审者注意的事项。如果 PR 包含 UI 更改，请附上截图或录屏。
    *   **允许维护者修改 (可选)**：勾选 "Allow edits by maintainers" 通常有助于维护者快速修复小问题。
    *   **提交 PR**：点击 "Create pull request"。

## Pull Request 评审

*   提交 PR 后，项目的 CI/CD 流程会自动运行测试和检查。
*   项目维护者会评审您的代码和文档，并可能提出修改意见。
*   请及时回应评审意见并进行必要的修改。维护者可能会直接在您的分支上进行小的修订（如果您允许的话），或者您需要自己更新代码并重新推送。
*   一旦 PR 被批准并通过所有检查，维护者会将其合并到主分支中。

---

再次感谢您对 Torch-RecHub 的贡献！