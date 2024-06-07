# Tuft Dental Database Segmentation Experiment

This repository contains experiments on the Tuft Dental Datasets using various state-of-the-art segmentation architectures. The goal is to compare the performance of different models in segmenting dental images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Backbones](#backbones)
- [Hyperparameters](#hyperparameters)
- [Criterion and Evaluation Metrics](#criterion-and-evaluation-metrics)
- [Installation](#installation)
- [Libraries Usage](#libraries-usage)

## Introduction
In this project, we explore the segmentation of dental images using several advanced neural network architectures. The models include U-Net, FPN, PAN, PSPNet, DeepLabV3, DeepLabV3+, U-Net++, and variants with ResNet34 and MobileNetV2 backbones. The objective is to identify the best performing model for accurate and efficient dental image segmentation.

## Dataset
The **Tuft Dental Database** is a valuable resource for dental diagnostics, featuring panoramic X-ray images that offer a comprehensive view of the upper and lower jaws. These images capture detailed structures including teeth, jawbones, and surrounding areas, making them essential for identifying issues like impacted teeth, jaw disorders, and assessing overall dental health. The dataset is meticulously annotated to support the development and evaluation of advanced segmentation models.

### Dataset Split
- **Training Set**: 700 images
- **Validation Set**: 150 images
- **Test Set**: 150 images

To enhance the dataset's robustness, a variety of transformations from the Albumentations library have been applied. These augmentations are thoughtfully executed to maintain the intricate details and integrity of the original images, ensuring high-quality data for model training and testing.

The dataset has been augmented using various transformations from Albumentations. These augmentations are applied carefully to ensure that the detailed information within the images is preserved.
## Models
The following segmentation architectures are implemented and compared:
- **U-Net**: 
  - **Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
  - **Description**: A convolutional network architecture for fast and precise segmentation of images. It uses a U-shaped architecture with skip connections for accurate localization and context capture.

- **FPN (Feature Pyramid Networks)**: 
  - **Paper**: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
  - **Description**: A feature pyramid network for building high-level feature maps at different scales. It enhances the capability of the network to detect objects at various scales.

- **PAN (Pyramid Attention Network)**: 
  - **Paper**: [Pyramid Attention Network for Semantic Segmentation](https://arxiv.org/abs/1805.10180)
  - **Description**: A network that enhances feature representation by integrating local and global context information through pyramid attention modules, improving semantic segmentation accuracy.

- **PSPNet (Pyramid Scene Parsing Network)**: 
  - **Paper**: [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
  - **Description**: Utilizes pyramid parsing modules to aggregate context information. It captures global context information to improve pixel-level segmentation.

- **DeepLabV3**: 
  - **Paper**: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
  - **Description**: Employs atrous convolutions for multi-scale context aggregation. It allows for dense feature extraction without losing resolution.

- **DeepLabV3+**: 
  - **Paper**: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
  - **Description**: An extension of DeepLabV3 that includes an encoder-decoder structure for better performance. It combines the benefits of spatial pyramid pooling and encoder-decoder architectures.

- **U-Net++**: 
  - **Paper**: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
  - **Description**: A nested U-Net architecture designed for more precise segmentation. It introduces dense skip connections for better feature fusion.

## Backbones
The following backbones are used for feature extraction in some of the models:
- **ResNet34**: 
  - **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - **Description**: ResNet34 is a 34-layer residual network that uses skip connections to jump over some layers, helping to mitigate the vanishing gradient problem. This architecture allows for training very deep networks, enabling significant performance improvements on various benchmarks. ResNet34 is known for its simplicity and scalability.

- **MobileNetV2**: 
  - **Paper**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
  - **Description**: MobileNetV2 is designed for mobile and embedded vision applications, focusing on computational efficiency and low memory usage. It introduces inverted residuals, where the input and output are thin bottleneck layers, and linear bottlenecks to maintain a rich feature representation. MobileNetV2 provides a good balance between accuracy and efficiency.

## Hyperparameters

The following hyperparameters are used for training the models:

- **Input Size**: 256 x 512
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Scheduler**: Reduce On Plateau
  - **Factor**: 0.5
  - **Patience**: 5
- **Max Epochs**: 300
- **Early Stopping**: Patience 20


## Criterion and Evaluation Metrics
### Criterion
We use `DiceLoss` as the loss function during training to improve the model's performance on segmentation tasks.
### Evaluation Metrics

The performance of each model is evaluated using the following metrics:
- Dice Coefficient: Measures the overlap between the predicted segmentation and the ground truth. It is particularly useful for imbalanced datasets.
- Pixel Accuracy: Computes the percentage of correctly classified pixels in the entire image.
- IoU (Intersection over Union): Measures the intersection between the predicted segmentation and the ground truth divided by their union, providing a robust evaluation of segmentation performance.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/siapai/tuft-dental-segmentation.git
cd tuft-dental-segmentation
pip install -r requirements.txt
```

## Libraries Usage

The following libraries are utilized in this project to ensure efficient and effective processing and modeling of the dataset:

- **Torch**: `2.3.0`
  - The core library for building and training neural networks in PyTorch.

- **Torchvision**: `0.18.0`
  - A package containing popular datasets, model architectures, and common image transformations for computer vision tasks.

- **Albumentations**: `1.4.8`
  - Used for advanced data augmentation techniques to enhance the robustness of the model without losing important details in the images.

- **PyTorch Lightning**: `2.2.4`
  - Provides a high-level framework for organizing and managing PyTorch code, enabling easier experimentation and model training.

- **Torchmetrics**: `1.4.0`
  - Offers a wide range of metrics for evaluating machine learning models in PyTorch, facilitating consistent and comprehensive performance assessment.

- **TensorBoard**: `2.16.2`
  - A tool for visualizing and monitoring the training process, providing insights into model performance and helping with debugging and optimization.




