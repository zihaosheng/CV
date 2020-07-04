# 基于PyTorch&YOLOv4的口罩佩戴检测

本项目是利用YOLOv4进行口罩佩戴检测，使用PyTorch实现。


## 目录
- [基于PyTorch&YOLOv4的口罩佩戴检测](#基于pytorchyolov4的口罩佩戴检测)
  - [目录](#目录)
  - [安装](#安装)
    - [数据集](#数据集)
    - [依赖库](#依赖库)
    - [模型权重](#模型权重)
  - [使用](#使用)
    - [检测图片](#检测图片)
    - [检测视频](#检测视频)
    - [训练](#训练)
    - [评估](#评估)
  - [参考](#参考)

## 安装
### 数据集
  - 初始数据集[链接](https://pan.baidu.com/s/1v06PLSN52YYJJyzBMhWJBQ)  提取码：31f3
  - 黑色口罩数据集[链接](https://pan.baidu.com/s/1nYsKzWFENpkKPkauEQxn1Q)  提取码：9hvg
### 依赖库
- Python >= 3.7
- PyTorch >= 1.4.0
- opencv-python >= 4.2.0.32
- Pillow >= 7.0.0
### 模型权重
  -  COCO数据集预训练模型：[链接](https://pan.baidu.com/s/1JDclXgxDmA06Mv6hrRB5Sw)  提取码：cp4g
  - 本项目冻结训练权重：[链接](https://pan.baidu.com/s/1Y_3EbdSEQvuNPwmovky9hg)  提取码：wm17
  - 本项目解冻训练权重：[链接](https://pan.baidu.com/s/1Kl1bC0iwEurL-p3WUkugtA)  提取码：dhzs
## 使用
### 检测图片
使用Jupyter Notebook打开predict.ipynb，设置好图片路径后，运行detect_image()函数即可。

### 检测视频
使用Jupyter Notebook打开predict.ipynb，设置好视频路径后，运行detect_video()函数即可。

### 训练
下载预训练模型：
  -  COCO数据集预训练模型：[链接](https://pan.baidu.com/s/1JDclXgxDmA06Mv6hrRB5Sw)  提取码：cp4g
  - 本项目冻结训练权重：[链接](https://pan.baidu.com/s/1Y_3EbdSEQvuNPwmovky9hg)  提取码：wm17
  - 本项目解冻训练权重：[链接](https://pan.baidu.com/s/1Kl1bC0iwEurL-p3WUkugtA)  提取码：dhzs

使用Jupyter Notebook打开train.ipynb，设置好数据路径、模型路径以及超参数后，即可进行训练。

### 评估
使用Jupyter Notebook打开eval.ipynb，设置好测试集路径后，运行即可生成detection-results和ground-truth。

再运行mAP目录下的main.py，即可计算mAP等结果。
  
## 其他
### 黑色口罩爬虫
### 训练&评估结果
## 参考
- 部分数据集来源：
  - https://github.com/hikariming/virus-mask-dataset
  - https://www.kesci.com/home/dataset/5e958c69e7ec38002d033362
- YOLOv4 PyTorch基于：https://github.com/Bil369/MaskDetect-YOLOv4-PyTorch

