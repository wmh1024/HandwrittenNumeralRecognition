# 基于Pytorch的手写数字识别系统

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/wmh1024/HandwrittenNumeralRecognition">
    <img src="img/icon.png" alt="Logo" width="100" height="100" style="border-radius: 30px;">
  </a>

<h3 align="center">基于Pytorch的手写数字识别</h3>
  <p align="center">
    实现了基于MNIST训练数据集版本和自制数据集版本的手写数字识别。
    <br />
    对CUDA和MPS做了对应优化，提高了识别准确度和速度。
    <br />
</p>

## 上手指南

克隆此项目

```sh
git clone https://github.com/wmh1024/HandwrittenNumeralRecognition.git
```

基于MNIST训练数据集版本

```sh
python DataByMNIST.py
```

基于自制数据集版本

```sh
python DataByMyData.py
```

## 使用到的框架

- pytorch
- matplotlib
- numpy

## 训练结果

测试设备：MacBook Air 2020 M1 8G内存

### 对于MNIST训练数据集

|    版本    |   准确率   |  训练时间   | epoch |
|:--------:|:-------:|:-------:|:-----:|
| MNIST数据集 | 99.170% | 155.95s |  10   |

### 对于自制数据集

|  版本   |   准确率   |  训练时间  | epoch |
|:-----:|:-------:|:------:|:-----:|
| 自制数据集 | 58.065% | 28.77s |  20   |

### 对于MPS优化（基于MNIST数据集）

| 版本  |   准确率   |  训练时间   | epoch |
|:---:|:-------:|:-------:|:-----:|
| CPU | 99.180% | 313.33s |  10   |
| MPS | 99.170% | 155.95s |  10   |

> 测试结果仅供参考，具体结果请根据自己的数据集进行测试。

## 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 版本控制

该项目使用 Git 进行版本管理。您可以在`repository`参看当前可用版本。

## 作者

[wmh](https://github.com/wmh1024)、[B1ld](https://github.com/z1922569567)、[伊伊得一](https://gitee.com/yide-yi)、[好想吃烤鱼饭](https://gitee.com/yan-mengjie1)

*您也可以在贡献者名单中参看其他参与该项目的开发者。*

## 版权说明

该项目签署了MIT授权许可，详情请参阅 [LICENSE.txt](https://github.com/wmh1024/HandwrittenNumeralRecognition/blob/main/LICENSE.txt)

<!-- links -->

[your-project-path]:wmh1024/HandwrittenNumeralRecognition

[contributors-shield]: https://img.shields.io/github/contributors/wmh1024/HandwrittenNumeralRecognition.svg?style=flat-square

[contributors-url]: https://github.com/wmh1024/HandwrittenNumeralRecognition/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/wmh1024/HandwrittenNumeralRecognition.svg?style=flat-square

[forks-url]: https://github.com/wmh1024/HandwrittenNumeralRecognition/network/members

[stars-shield]: https://img.shields.io/github/stars/wmh1024/HandwrittenNumeralRecognition.svg?style=flat-square

[stars-url]: https://github.com/wmh1024/HandwrittenNumeralRecognition/stargazers

[issues-shield]: https://img.shields.io/github/issues/wmh1024/HandwrittenNumeralRecognition.svg?style=flat-square

[issues-url]: https://img.shields.io/github/issues/wmh1024/HandwrittenNumeralRecognition.svg

[license-shield]: https://img.shields.io/github/license/wmh1024/HandwrittenNumeralRecognition.svg?style=flat-square

[license-url]: https://github.com/wmh1024/HandwrittenNumeralRecognition/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/shaojintian