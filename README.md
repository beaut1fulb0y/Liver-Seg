# Liver-Seg

## 项目安装

- 将项目克隆到目录中

- 数据集使用的是LiTS数据集，自行下载

- 将数据集解压缩，将2d文件夹放置在项目的根目录下

- 依赖库（可能不完整）
    - torch
    - torchvision
    - tensorboard
    - numpy
    - tqdm

## 项目介绍

训练可以通过命令行执行：

```bash
python main.py
```

如果有cuda和cudnn，那么可以加参数：

```bash
python main.py -d cuda
```

已经训练好的模型参数是torch.state_dict，在`params/`文件夹中

`runs/`中保存了分割测试结果，以及tensorboard日志的压缩包文件，可以解压缩以后，使用命令行：

```bash
tensorboard --logdir >path-to-logdir
```

然后给定的网页端口查看训练日志。

项目中除了使用深度学习，也使用了CV的方法，

## 结果

通过运行`test.py`生成分割二值图像，自动放在`runs/`中，`calc.py`可以计算所有的要求测试参数，并且绘制ROC曲线并保存在项目根目录下

作者：宸哥

github链接：https://github.com/beaut1fulb0y/Liver-Seg.git
