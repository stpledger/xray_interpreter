# 🩻X-ray Diagnostic Model

This repository is an effort to create a multi-class classification vision model to identify the 14 diagnoses in the [NIH Chest X-ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data). 

## 📌 Objective

Fine-tune a pre-trained vision model to identify the presence of any of the 14 diagnoses with the highest accuracy. Specifically, a high F1-score is most desirable since many diagnoses are only present in ~1% of the dataset. The authors of the dataset also explain how the labels were created using NLP extraction from medical notes, and they state that the labels are not perfect but are expected to be >90% accuracy.

## 🧠 Approach Overview

This notebook adopts a deep learning-based approach, primarily by fine-tuning EfficientNet variants on the task. A custom preprocessing pipeline was built to scale the images to a resolution compatible with the model of choice and then augment the images to increase variance and reduce the risk of overfitting. Then, I experimented with varying methods such as class reweighting to counteract the severely imbalanced dataset.

## 🛠️ Key Components

### 🧹 Preprocessing

The following transforms and augmentations are applied to the images:
1. Rescale the image to proper resolution (square -> square, so no stretching or cropping required)
2. Convert the image to RGB (3 channels) if it's in another format
3. Add color jitter (randomness in brightness, contrast, saturation, and hue) to reduce overfitting risk
4. Add a small random rotation to reduce overfitting risk

### 🧮 Modeling

- **EfficientNet**: I chose to use EfficientNet as a starting model because its compact size could fit entirely on my personal GPU, and I wanted to explore full fine-tuning options before exploring other techniques. Additionally, after running some experiments, it became very clear that the using the pre-trained weights as opposed to random initialization made a large difference in the speed of the model's convergence.

### ⚙️ Training Configuration

- Optimizer: `AdamW`  
- Learning Rate Scheduler: Custom Warmup + Cosine Annealing schedule
- Loss Function: `BCEWithLogitsLoss`
- Evaluation Metric: Accuracy, Precision, Recall, F1 Score

### ✅ Validation Strategy

A standard randomized train-test split is used, and the ultimate performance metrics are based on the model's performance on the validation dataset.

## 📈 Results & Insights

- While the pre-trained EfficientNet certainly demonstrates stronger performance compared to a newly initialized network, it still shows limited performance with an F1 score of ~0.38, while still developing accuracy of 95%.
- Rescaling the images could remove crucial information, making it difficult for the network to identify commonalities between positive examples.
- The next step would be to use a larger pre-trained neural network with LoRA and mixed-precision training. Additionally, removing the data augmentation steps until overfitting is demonstrable could lead to quicker experiments.

## 🧾 File Structure

- [`scratchpad.ipynb`](scratchpad.ipynb): Jupyter notebook for ad hoc tasks like train-test split and viewing post-processed images.
- [`train.py`](train.py): The custom Pytorch Lightning Trainer and training loop
- [`model.py`](model.py): Initialization of the pre-trained EfficientNet model variants
- [`data.py`](data.py): Defines the ImageDataset class to handle cacheing. Used by the DataLoader
- [`metrics.py`](metrics.py): Custom metrics like a class-wise confusion matrix used to calculate class-wise precision, recall, and F1 scores
- [`transforms.py`](transforms.py): Image preprocessing and augmentation functions