# Support Documentation

Welcome to Torch-RecHub! This document will help you get the support you need and answer common questions.

## 🆘 Getting Help

### 1. 📚 Check Documentation

Before seeking help, please check our documentation:

- **[README](README.md)**: Project overview and quick start
- **[API Documentation](docs/)**: Detailed API reference
- **[Tutorials](tutorials/)**: Jupyter tutorials and examples
- **[Example Code](examples/)**: Usage examples for various models
- **[FAQ](#-frequently-asked-questions)**: FAQ section in this document

### 2. 🔍 Search Existing Issues

Before creating a new issue, please search [existing issues](https://github.com/datawhalechina/torch-rechub/issues):

- Use keywords to search for related problems
- Check closed issues
- Look for similar feature requests

### 3. 💬 Community Discussion

- **GitHub Discussions**: For general discussions and Q&A
- **Issues**: For bug reports and feature requests

### 4. 📧 Direct Contact

For urgent issues or private consultations, contact project maintainers:

- **Project Lead**: [morningsky](https://github.com/morningsky)
- **GitHub Issues**: [Create new issue](https://github.com/datawhalechina/torch-rechub/issues/new)

## 🐛 Reporting Bugs

If you found a bug, please follow these steps to report it:

### 1. Confirm It's a Bug

- Check if your code is correct
- Confirm the issue is reproducible
- Check if there are existing reports

### 2. Gather Information

Please provide the following information:

```
**Environment Information:**
- Operating System: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.8.10]
- PyTorch Version: [e.g., 1.12.0]
- Torch-RecHub Version: [e.g., 0.0.3]
- CUDA Version (if applicable): [e.g., 11.3]

**Problem Description:**
[Clear description of the issue encountered]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [Third step]

**Expected Behavior:**
[Describe what you expected to happen]

**Actual Behavior:**
[Describe what actually happened]

**Error Messages:**
```
[Paste complete error messages and stack traces]
```

**Minimal Reproduction Example:**
```python
# Provide minimal code example that reproduces the issue
```
```

### 3. Create Issue

Use our [Bug Report Template](https://github.com/datawhalechina/torch-rechub/issues/new?template=bug_report.md) to create an issue.

## 💡 Feature Requests

We welcome feature suggestions! Please submit them as follows:

### 1. Describe the Need

- **Problem Background**: Describe the problem or need you encountered
- **Proposed Solution**: Present your solution ideas
- **Alternative Solutions**: Consider other possible solutions
- **Use Cases**: Explain the application scenarios for the feature

### 2. Create Feature Request

Use our [Feature Request Template](https://github.com/datawhalechina/torch-rechub/issues/new?template=feature_request.md) to create an issue.

## ❓ Frequently Asked Questions

### Installation Issues

**Q: What to do when encountering dependency conflicts during installation?**

A: We recommend using a virtual environment:
```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Then install
uv sync
```

**Q: How to install GPU version of PyTorch?**

A: Please refer to the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) to choose the version suitable for your system.

### Usage Issues

**Q: How to load custom datasets?**

A: Please refer to examples in the `examples/` directory or check tutorials in `tutorials/`.

**Q: What to do when running out of GPU memory during training?**

A: Try the following solutions:
- Reduce batch_size
- Use gradient accumulation
- Enable mixed precision training
- Use a smaller model

**Q: How to save and load models?**

A: Use PyTorch's standard methods:
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))
```

**Q: How to customize loss functions?**

A: Inherit from `torch.nn.Module` and implement the `forward` method:
```python
class CustomLoss(torch.nn.Module):
    def forward(self, pred, target):
        # Implement your loss function
        return loss
```

### Performance Issues

**Q: What to do when training is very slow?**

A: Check the following aspects:
- Ensure GPU usage (if available)
- Optimize data loading (increase num_workers)
- Use appropriate batch_size
- Consider using mixed precision training

**Q: How to perform model tuning?**

A: Recommended tuning steps:
1. Start validation with small datasets
2. Adjust learning rate
3. Try different optimizers
4. Adjust model architecture parameters
5. Use learning rate schedulers

### Development Issues

**Q: How to add new models?**

A: Please refer to the detailed instructions in [Contributing Guide](CONTRIBUTING.md).

**Q: How to run tests?**

A: Use the following commands:
```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/test_specific.py
```

## 📈 性能优化建议

### 数据处理优化

- 使用 `DataLoader` 的 `num_workers` 参数
- 预处理数据并缓存
- 使用适当的数据类型（如 `float32` 而非 `float64`）

### 训练优化

- 使用 GPU 加速
- 启用混合精度训练（AMP）
- 合理设置 batch_size
- 使用梯度累积处理大 batch

### 内存优化

- 及时释放不需要的变量
- 使用 `torch.no_grad()` 在推理时
- 考虑使用检查点技术

## 🔄 版本兼容性

### 支持的版本

- **Python**: 3.8+
- **PyTorch**: 1.7+
- **NumPy**: 1.19+
- **Pandas**: 1.2+

### 升级指南

在升级版本时：

1. 查看 [发布说明](https://github.com/datawhalechina/torch-rechub/releases)
2. 检查破坏性变更
3. 更新您的代码
4. 运行测试确保兼容性

## 📞 紧急支持

对于紧急问题（如安全漏洞），请：

1. **不要**在公开 Issue 中报告安全问题
2. 发送邮件给项目维护者
3. 提供详细的问题描述
4. 我们会在 48 小时内回复

## 🤝 社区支持

我们鼓励社区成员互相帮助：

- 回答其他用户的问题
- 分享使用经验和技巧
- 贡献示例代码和教程
- 参与代码审查

## 📝 反馈

您的反馈对我们很重要：

- **功能建议**：告诉我们您需要什么功能
- **使用体验**：分享您的使用感受
- **文档改进**：指出文档中的问题
- **性能反馈**：报告性能问题

---

感谢您使用 Torch-RecHub！我们致力于为您提供最好的支持。🚀