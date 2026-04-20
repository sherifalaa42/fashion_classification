Overview

This project is an end-to-end deep learning pipeline for classifying fashion images. It leverages transfer learning with MobileNetV2 to achieve strong performance while remaining lightweight enough for deployment in a cloud-based environment.

The model is optimized for real-world usage and deployed using a serverless architecture on AWS.

Features
Image classification using deep learning (TensorFlow/Keras)
Transfer learning with MobileNetV2 (pre-trained on ImageNet)
Data augmentation for improved generalization
Lightweight model optimized with TensorFlow Lite (TFLite)
REST API built with Flask
Containerized using Docker
Serverless deployment with AWS Lambda (free tier)

Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet)
Framework: TensorFlow / Keras
Data Preprocessing: ImageDataGenerator (augmentation)
Optimizer: Adam
Loss Function: Categorical Crossentropy

Performance
Validation Accuracy: 86%

Tech Stack
Python
TensorFlow / Keras
Flask
Docker
AWS Lambda
TensorFlow Lite (TFLite)

Project Workflow
Data preprocessing and augmentation using ImageDataGenerator
Model training using MobileNetV2 with transfer learning
Model evaluation (achieved 86% validation accuracy)
Model optimization using TensorFlow Lite
API development using Flask
Containerization using Docker
Deployment using AWS Lambda (serverless)

Deployment

The model is deployed using a serverless architecture:

Flask API wrapped inside a Docker container
Integrated with AWS Lambda for scalable inference
Hosted on AWS free tier
