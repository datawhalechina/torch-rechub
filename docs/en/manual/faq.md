---
title: Frequently Asked Questions
description: Common questions and troubleshooting guide for Torch-RecHub
---

# Frequently Asked Questions

Common questions and troubleshooting guide for Torch-RecHub.

* **Will there be a TensorFlow version?**

    - Not currently planned
  
    - This project's core positioning is to provide easy-to-use model implementations for beginners, referencing industry-used models. PyTorch has a wider audience base.

* **Why is the AUC=0 when running the example?**

    - The example contains 100 sample data entries, which are meant to demonstrate data format and feature types, ensuring the code runs smoothly. It does not guarantee performance metrics.
    - If you need to test performance, you can download the dataset using the links described in the readme, then refer to the parameter configuration file in the example for model training and evaluation.

* **Installing annoy**

    - Installing annoy on Windows

        - Online installation

        ```Bash
        pip install annoy
        ```

        If you don't have C++ related compilation environment on Windows, you might see the following error:

        ```Bash
        error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
        ```

        Error screenshot:
        ![alt text](/img/win_install_annoy_error.png "Error screenshot")
        
        In this case, you can use offline installation

        - Offline installation

        Download annoy library from: [https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

        ```Bash
        pip install annoy‑1.17.0‑cp39‑cp39‑win_amd64.whl
        ```

    - Installing annoy on Linux/Mac OS

        - Online installation

        ```text
        pip install annoy
        ```

        Normally Mac can install successfully online. If online installation fails with nose-related errors at the bottom, proceed with offline compilation installation

        - Offline installation

          - Download nose

            Download from: [https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_annoy)

            ```Bash
            pip install nose‑1.3.7‑py3‑none‑any.whl
            ```

          - Download annoy

            Download from: [https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz](https://files.pythonhosted.org/packages/a1/5b/1c22129f608b3f438713b91cd880dc681d747a860afe3e8e0af86e921942/annoy-1.17.0.tar.gz)

            ```Bash
            tar -zxvf annoy-1.17.0.tar.gz
            cd annoy-1.17.0
            python setup.py install
            ```

      After installing annoy, you can proceed to install torch-rechub

      ```Python
      pip install --upgrade torch-rechub
      ```

