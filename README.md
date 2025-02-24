# Knife Image Classification using Deep Learning

This repository contains a deep learning-based system designed to classify knife images into 192 distinct categories. The project was developed as coursework for the **Fundamentals of Machine Learning (EEEM066)** module at the **University of Surrey**.

## 🔍 Project Overview
Knife-related crime has risen significantly, prompting a need for automated weapon identification systems. This project addresses that need by employing advanced convolutional neural networks (CNNs) to accurately classify knives from images.

## 📚 Dataset
- Consists of approximately **9,928 knife images** divided into **192 classes**.
- An additional test dataset of **351 images** was used for model evaluation.
- Dataset is privately provided for educational purposes and cannot be redistributed.

## 🛠️ Methods and Techniques
- Explored CNN architectures: **EfficientNet-B0**, **ResNet50**, and **DenseNet121**.
- Applied extensive **data augmentation techniques** including random rotation, horizontal flipping, brightness, contrast, saturation, and hue adjustments.
- Conducted systematic **hyperparameter tuning** (learning rate, batch size) to optimize performance.
- Model evaluated using the **mean Average Precision (mAP)** metric.

## 📈 Results
| Model            | Best Epoch | Train mAP | Validation mAP |
|------------------|------------|-----------|----------------|
| EfficientNet-B0  | 12         | 0.567     | 0.592          |
| ResNet50         | 4          | 0.677     | **0.690**      |
| DenseNet121      | 7          | 0.593     | 0.604          |

- Best performance achieved by **ResNet50** with a validation mAP of **0.690**.
- Augmentation with **Random Rotation + Horizontal Flip** improved mAP to **0.717**.

## 🚀 Quick Start

### Requirements
Install the dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train the model:
```bash
bash train.sh
```

### Evaluation
To evaluate the trained model:
```bash
bash test.sh
```




### ⚙️ Key Dependencies
- Python 3.x
- PyTorch
- timm
- OpenCV
- pandas
- scikit-learn



