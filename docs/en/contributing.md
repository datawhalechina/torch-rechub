# Contribution Guide

Thank you very much for your interest in the Torch-RecHub project and for considering contributing! Your help is crucial for the development of the project. This guide will detail how to participate in contributions.

## How to Contribute

You can contribute in the following ways:

1.  **Report Bugs**: If you find any errors or unexpected behavior while using Torch-RecHub, please submit an Issue detailing the problem, steps to reproduce it, and your environment information.
2.  **Suggest Enhancements**: If you have ideas for improving existing features or adding new ones, please create an Issue first to discuss.
3.  **Submit Code Changes**: Fix bugs, implement new features, or improve code quality.
4.  **Improve Documentation**: Enhance or correct documentation, write tutorials, or provide example use cases.

## Finding Contribution Points

You can start from the following aspects:

1.  **Check Issues**: Browse the [project Issues list](https://github.com/datawhalechina/torch-rechub/issues), look for issues marked with `help wanted`.
2.  **Improve Existing Features**: If you find areas for optimization during use, feel free to propose suggestions or submit improvements directly.
3.  **Implement New Features**: If you have new ideas, it is recommended to create an Issue for discussion first to ensure it aligns with the project direction.

## Contribution Process (Code and Documentation)

We use the standard GitHub Fork & Pull Request workflow to accept code and documentation contributions. (You can also perform the following operations on the GitHub website).

1.  **Fork the Repository**
    Visit the Torch-RecHub [GitHub repository page](https://github.com/datawhalechina/torch-rechub), click the "Fork" button in the top right corner to copy the project to your own GitHub account.

2.  **Clone Your Fork**
    Clone your forked repository locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/torch-rechub.git
    cd torch-rechub
    ```
    Please replace `YOUR_USERNAME` with your GitHub username.

3.  **Set Upstream Remote (Optional but Recommended)**
    Add the original project repository as an upstream remote for easy synchronization of updates:
    ```bash
    git remote add upstream https://github.com/datawhalechina/torch-rechub.git
    # Sync updates from upstream main branch if needed
    # git fetch upstream
    # git checkout main
    # git merge upstream/main
    ```

4.  **Make Changes**
    Write code, modify documentation, or make other improvements directly on the `main` branch.

5.  **Ensure Code Quality**
    *   **Code Style**: Please adhere to the project's existing code style.
    *   **Documentation**: For user-visible changes (like new features, API changes), please update relevant documentation (README, files under `docs/`, etc.).

6.  **Commit Changes**
    Commit your changes with clear and meaningful commit messages. We recommend following the [Conventional Commits](https://www.conventionalcommits.org/) specification.
    ```bash
    git add .
    git commit -m "feat: Add support for XXX model"
    # Or
    git commit -m "fix: Correct typo in README"
    # Or
    git commit -m "docs: Update contribution guide"
    ```

7.  **Push Branch**
    Push your local `main` branch to your forked GitHub repository:
    ```bash
    git push origin main
    ```

8.  **Create a Pull Request (PR)**
    Return to your forked repository page on GitHub. You should see a prompt suggesting you create a Pull Request based on the recent pushes to `main`. Click that prompt or manually navigate to the "Pull requests" tab and click "New pull request".
    *   **Choose Branches**: Ensure the Base repository is `datawhalechina/torch-rechub`'s `main` branch, and the Head repository is your fork's `main` branch.
    *   **Fill in PR Information**:
        *   **Title**: Concisely describe the purpose of the PR, often based on the commit message.
        *   **Description**: Detail the changes you've made, the problem solved (you can link related Issues, e.g., `Closes #123`), and any points reviewers should note. If the PR includes UI changes, please attach screenshots or screen recordings.
    *   **Allow Maintainer Edits (Optional)**: Checking "Allow edits by maintainers" often helps maintainers quickly fix minor issues.
    *   **Submit PR**: Click "Create pull request".

## Pull Request Review

*   After submitting the PR, the project's CI/CD workflow will automatically run tests and checks.
*   Project maintainers will review your code and documentation and may suggest modifications.
*   Please respond promptly to review comments and make necessary changes. Maintainers might make minor edits directly on your branch (if you allow it), or you might need to update the code yourself and push again.
*   Once the PR is approved and passes all checks, maintainers will merge it into the main branch.

---

Thank you again for your contribution to Torch-RecHub!