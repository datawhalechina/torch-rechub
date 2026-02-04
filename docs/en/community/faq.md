---
title: FAQ
description: Torch-RecHub frequently asked questions and troubleshooting guide
---

# FAQ

Torch-RecHub frequently asked questions and troubleshooting guide.

* **Will there be a TensorFlow version?**

    - Not currently planned

    - This project's core positioning is to provide easy-to-use model implementations for beginners, referencing industry-used models. PyTorch has a wider audience.

* **Why does running the example give AUC=0?**

    - The examples use 100 sample records for users to reference data format and feature types, ensuring the code runs smoothly. Accuracy is not guaranteed.
    - If you need to test performance, download the data from the links described in the README, then refer to the parameter configuration files in the examples for model training and evaluation.

- **Annoy Installation**

    - Windows installation

        - Online installation

        ```bash
        pip install annoy
        ```

        If Windows doesn't have a C++ compilation environment, you may see this error:

        ```bash
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        ```

        In this case, use offline installation:

        - Offline installation

        Annoy library download: [https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

        ```bash
        pip install annoy‑1.17.0‑cp39‑cp39‑win_amd64.whl
        ```

    - Linux/macOS installation

        - Online installation

        ```bash
        pip install annoy
        ```

        Normally macOS can install online successfully. If online installation fails with a nose-related error, use offline compilation:

        - Offline installation

          - Download nose

            Download: [https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

            ```bash
            pip install nose‑1.3.7‑py3‑none‑any.whl
            ```

          - Download annoy

            Download: [https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz](https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz)

            ```bash
            tar -zxvf annoy-1.17.0.tar.gz
            cd annoy-1.17.0
            python setup.py install
            ```

      After installing annoy, you can install torch-rechub:

      ```bash
      pip install --upgrade torch-rechub
      ```

