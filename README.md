# Free Space Segmentation Model

## Introduction
The Free Space Segmentation Model is a cutting-edge solution designed to enhance the capabilities of autonomous systems in navigating complex environments. Utilizing the HybridNets multitask architecture, this model effectively identifies and segments drivable areas and free spaces in real-time, providing critical information for decision-making in autonomous driving, robotics, and urban planning applications.

In an era where smart mobility and automation are at the forefront of technological advancement, accurate perception of the environment is essential. This model leverages the ONNX format to ensure compatibility and efficiency across various platforms, making it suitable for both research and practical deployment. By integrating this segmentation model into your projects, you can significantly improve the performance and safety of autonomous navigation systems, enabling them to adapt to diverse and dynamic scenarios.


## Table of Contents
- [Model_Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Inference](#model-inference)

## Model Architecture
The HybridNets model architecture is designed for multitask automotive applications, integrating EfficientNet-B0 as the backbone for feature extraction. It employs a BiFPN (Bidirectional Feature Pyramid Network) to combine features from different scales, enhancing both detection and segmentation. 

**1. Backbone(EfficientNet-B0) :** EfficientNet-B0 serves as the backbone. It's known for its balance between accuracy and efficiency, using compound scaling to optimize depth, width, and resolution.
This backbone is lightweight, enabling faster inference times while maintaining high accuracy.

**2. BiFPN (Bidirectional Feature Pyramid Network) :**
BiFPN enables the fusion of multi-scale features, allowing the network to efficiently combine low-level and high-level information. It supports bidirectional cross-scale connections for more robust feature extraction.
It enhances object detection, segmentation, and lane detection by merging features from different layers, making it easier for the model to process varying input resolutions.

**3. Drivable Area Segmentation Head :** Performs pixel-wise segmentation to classify the road and non-road areas, crucial for autonomous driving.

**4. Lane Detection Head :** Identifies lane markings on the road, helping vehicles stay in lanes.

**5. Postprocessing:** The outputs from these heads are post-processed to refine the segmentation maps and object detection results. This includes techniques like non-maximum suppression for object detection and upsampling for segmentation.
### **Training Strategy**
**Datasets:** The model is trained on datasets like BDD100K for object detection, lane detection, and segmentation tasks.

**Loss Functions :** HybridNets uses a combination of losses: cross-entropy for segmentation, focal loss for object detection, and specific losses for lane detection.

**Optimizations :** Techniques like mixed precision training, batch normalization, and augmentation are used to enhance the model’s performance and generalization.

**Output :** HybridNets outputs segmentation maps for drivable areas, lane markings, and bounding boxes for detected objects. This combination is especially useful in real-time applications like autonomous driving, where multiple tasks are handled simultaneously in a single forward pass.

![alt text](image.png)
## Features
- **Multitask Learning**: Utilizes the HybridNets architecture for simultaneous road segmentation and free space detection.
- **High Precision**: Achieves state-of-the-art performance in free space segmentation through advanced neural network techniques.
- **ONNX Compatibility**: Provides a streamlined workflow for deployment on various platforms using the ONNX format.
- **Real-time Processing**: Capable of processing video streams for dynamic environments.

##

To obtain the ONNX model for HybridNets, you can download it from Pinto's Model Zoo repository [**here**](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/276_HybridNets) include from here get onnx model. This repository contains pre-converted ONNX models along with scripts for optimizing and running them on different platforms. The page includes instructions on how to use the model and detailed files for deployment.

For direct access, navigate to the ONNX folder in the provided repository link, where you'll find the model ready for use.
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/krishnapriya-nynaru/FreeSpace-SegmentationModel.git
2. 
