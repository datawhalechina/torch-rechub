---
title: Contributing
description: Report a bug, suggest a feature or submit a pull request to Torch-RecHub
---

# Contributing

If an example fails, a tutorial is missing a step, or you want to add a model, open an issue or pull request. For a bug, include the example you ran, your Python and PyTorch versions, the command and the error message. That gives the next person a starting point for reproducing it. [Project issues](https://github.com/datawhalechina/torch-rechub/issues)

## Set up locally

Fork the repository on GitHub, replace `YOUR_USERNAME` below with your account name, then clone your fork and create a branch.

```bash
# Clone your fork and create a branch.
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

git switch -c pr/update-guide
```

Use the section below that matches your change.

## Edit the documentation

Chinese pages live in `docs/zh/` and their English counterparts in `docs/en/`. If you change a step or a parameter, update both languages. Run these commands from the repository root.

```bash
npm ci
npm run docs:build
```

Use `npm run docs:dev` for a live preview as you edit. Before submitting, open the changed pages and follow their links. If you updated an example, run the new steps and check that the output matches the description.

## Change Python code

Run `uv sync` from the repository root to install the development dependencies. If your change uses an optional feature, install its extra too. For example, `uv sync --extra onnx` sets up ONNX export.

When adding a model, start with an existing implementation for the same task. Include an example under `examples/` with the required features, the command to run and the expected output. For a bug fix, verify the change on a small sample that reproduces the problem, then run the relevant tests.

```bash
# Run all tests.
uv run pytest

# Run a specific test file.
uv run pytest tests/test_e2e_ranking.py

# Run the tests and generate a coverage report.
uv run pytest --cov=torch_rechub
```

Most tests run locally. To check the Milvus integration, start a local Milvus service and run the following in PowerShell.

```powershell
# PowerShell
$env:RUN_MILVUS_TESTS = "1"
uv run --extra milvus pytest tests/test_serving.py -m integration
```

Before submitting Python changes, run `uv run python config/format_code.py` to format the code, then inspect the diff.

## Submit a pull request

1. Summarize the change in a short commit message such as `docs: add the ranking example data path` or `fix: correct the sequence feature mask`.
2. Push to your fork with `git push -u origin pr/update-guide`, substituting your own branch name if needed.
3. Open a pull request against `main` in the upstream repository. Use the template to describe the problem, what you changed and the commands you used to verify it. Include the issue number if there is one.

Please follow the project's [Code of Conduct](https://github.com/datawhalechina/torch-rechub/blob/main/CODE_OF_CONDUCT.md) in discussions.
