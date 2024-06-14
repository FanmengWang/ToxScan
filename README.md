# A Multiscale-Information-Embedded Universal Toxicity Prediction Framework
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue)](https://github.com/FanmengWang/ToxScan/blob/master/LICENCE.txt)
[![Static Badge](https://img.shields.io/badge/PyTorch-red)](https://pytorch.org/)

[[Online Platform]](https://app.bohrium.dp.tech/toxscan/) 

This is the official implementation of "A Multiscale-Information-Embedded Universal Toxicity Prediction Framework"

ToxScan is a universal toxicity prediction framework to address the toxicity prediction issue.
<p align="center"><img src="figures/Overview.png" width=80%></p>
<p align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>An illustration of ToxScan and its workflow.</b></p>

Dependency
------------
 - [Uni-Core](https://github.com/dptech-corp/Uni-Core), you can check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
 - ```bash
   pip install rdkit-pypi==2021.9.4
   pip install dpdata
   ```

Data Preparation
------------
Please download the [dataset](https://drive.google.com/file/d/1TZT7pSuM_z8yHkVA9FOzUB2gwe1AiyGe/view?usp=drive_link) and place it to the fold `./dataset`, then
  ```bash
  cd dataset
  tar -xzvf dataset.tar.gz
  ```

Training
------------
  ```bash
  bash train.sh
  ```

Inference
------------
  ```bash
  bash inference.sh
  ```

Evaluation
------------
  ```bash
  python eval.py
  ```

Online Platform
------------
You can also try ToxScan online by clicking on this [link](https://app.bohrium.dp.tech/toxscan/)


Acknowledgment
--------
This code is built upon [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) and [Uni-Core](https://github.com/dptech-corp/Uni-Core). Thanks for their contribution.