# 自动发布

在 GitHub 页面选择 main 上准备发布的提交，填写版本标签（如 `v1.0.0`）和发布说明，点击 **Publish release**，无需提前手动修改版本号。

工作流会自动：

1. 从发布标签检出源码，更新 `pyproject.toml`、`uv.lock`、`CHANGELOG.md` 和中英文 API 文档。
2. 将发布元信息提交到 main，并触发文档部署。
3. 构建、检查 wheel/sdist，发布到 PyPI，并上传到 GitHub Release。

原始标签不会移动；包内容是在标签源码上补齐发布元信息后的结果。main 同步不使用强制推送；如果 main 已前进且无法快进，工作流会停止。

仓库需要保留原有的 PyPI 发布凭据，并允许 Actions 写入 main 和触发文档部署；分支保护规则仍然生效。
