# 3D Pose Estimation for Vehicles

## Introduction

This work generates 4 key-points and 2 key-edges from vertices and edges of vehicles as ground truth. The brightness of each generated heatmap is normal distribution. The model is combined with Resnet-50 for feature generation and three transpose-convolution layers for generation of heatmaps.

This code was adapted from an official pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. On COCO keypoints valid dataset, their best **single model** achieves **74.3 of mAP**. </br>

## Results
Training after 4 epochs (27000+ samples per epoch)
![image](https://raw.githubusercontent.com/RoboBachelor/vehicle_pose_estimation/master/%E6%95%88%E6%9E%9C.png)