---
title: 常见问题解答
description: Torch-RecHub 常见问题及故障排除指南
---

# 常见问题解答

Torch-RecHub 常见问题及故障排除指南

* **会推出 tensorflow 版本吗？**

    - 暂不考虑
  
    - 本项目核心定位是面向初学者提供容易上手的、业界使用的模型复现参考，pytorch 受众面较广

* **为什么跑 example 得出的 auc=0**

    - example 为 100 条示例数据，供用户参考数据格式、特征类型，保证代码畅通运行，不保证精度
    - 如果需要测试性能，可以按照 readme 中数据集描述的下载链接下载数据，然后参考 example 中的参数配置文件，进行模型训练与评估

- **annoy 安装**

    - windows 安装 annoy

        - 在线安装

        ```Bash
        pip install annoy
        ```

        如果 windows 上没有 C++相关编译环境，出现如下报错：

        ```Bash
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        ```

        报错截图：
        ![alt text](/img/win_install_annoy_error.png "报错截图")
        
        则可以采用离线安装方式

        - 离线安装

        annoy 库下载地址：[https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

        ```Bash
        pip install annoy‑1.17.0‑cp39‑cp39‑win_amd64.whl
        ```

    - linux/mac os 安装 annoy

        - 在线安装

        ```text
        pip install annoy
        ```

        正常 mac 可以在线安装成功，如果在线安装报错，最下方提示和 nose 相关报错，则进行离线编译安装

        - 离线安装

          - 下载 nose

            下载地址：[https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

            ```Bash
            pip install nose‑1.3.7‑py3‑none‑any.whl
            ```

          - 下载 annoy

            下载地址：[https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz](https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz)

            ```Bash
            tar -zxvf annoy-1.17.0.tar.gz
            cd annoy-1.17.0
            python setup.py install
            ```

      annoy 安装后，即可直接安装 torch-rechub 了

      ```Python
      pip install --upgrade torch-rechub
      ```

