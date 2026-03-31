# 无人机多模态检测数据集（RFVS: UAV Multi-Modal Detection Dataset）
A multi-modal UAV detection dataset integrating RGB images and RF signals. 一个融合视觉图像(RGB)与射频信号(RF IQ/STFT)的无人机多模态检测数据集，适用于复杂环境下的目标检测与分类研究。

欢迎来到 **RFVS** 官方代码与数据仓库！

随着低空经济与无人机技术的快速发展，复杂环境下的无人机安全监管与反制变得日益重要。传统的单一模态（纯视觉或纯雷达/射频）检测方法在面临视线遮挡、恶劣天气或复杂电磁环境时往往存在局限性。为此，本项目开源了 **RFVS** 这一融合视觉图像（RGB Images）与射频信号（RF IQ Data）的同步多模态无人机数据集。

本数据集不仅提供了丰富的无人机视觉图像及其精确的检测/分类标签，还**严格帧级同步**采集了对应时刻的底层射频原始 IQ 数据，并提供了将其转化为易于深度学习（CNN/ViT）模型处理的 STFT 时频图的标准代码。本项目的开源旨在为学术界和工业界提供一个高质量的基准测试平台，大力推动多模态融合、目标检测、细粒度分类以及射频信号分析等前沿深度学习任务的研究。

## 数据集结构
完整的数据集托管在网盘中，下载解压后，目录结构如下：
```
dataset/
├── images/            # 视觉图像数据 (RGB)
│   ├── test/
│   ├── train/
│   └── val/
├── labels/            # 目标检测标签 (格式: YOLO)
│   ├── test/
│   ├── train/
│   └── val/
├── labels-class/      # 细粒度分类标签数据
│   ├── test/
│   ├── train/
│   └── val/
└── RF_raw/            # 射频原始 IQ 采样数据 (二进制文件)
    ├── test/
    ├── train/
    └── val/
```

## 数据集下载
由于数据集体积较大，我们将完整数据托管在了云盘上。请通过以下链接下载：
- 百度网盘: 点击这里下载 (提取码: XXXX)
- Google Drive / Zenodo: 备用下载链接

下载后，请将数据集解压到项目根目录，或根据你的代码修改数据读取路径。

## 关键代码说明
本仓库提供 RFVS 数据集核心预处理脚本：`iq_to_stft.m`。

原始射频数据（RF_raw）为二进制 IQ 采样点，无法直接输入深度学习网络。我们提供 MATLAB 脚本，将 RF_raw 中的 IQ 数据进行短时傅里叶变换（STFT），生成无坐标轴的纯净二维时频图，保存至 `RF_images` 目录。

### 环境依赖
- MATLAB（推荐 R2020a 及以上版本）

### 使用方法
1. 克隆本仓库到本地：
```
git clone https://github.com/Stella050527/RFVS.git
```
2. 下载 RFVS 数据集并解压，确保 `RF_raw` 目录存在。
3. 在 MATLAB 中打开 `iq_to_stft.m`。
4. 确认采样率 Fs = 153.6e6 与 RFVS 实际数据匹配。
5. 运行脚本，自动解析 IQ 数据、生成 STFT 时频图并保存为 PNG。

## 引用 (Citation)
如果您在研究中使用了 **RFVS** 数据集或代码，请引用我们的工作：
```bibtex
@article{YourName2026RFVS,
  title={RFVS: A Synchronized Multimodal RF–Vision Dataset for Tiny UAV Detection and Classification},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## 许可证 (License)
RFVS 数据集及代码遵循 **MIT License** 开源协议。
