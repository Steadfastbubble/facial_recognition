# 👶👴 Age Regression with CNNs: Predicting Age from Facial Images

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338E?style=for-the-badge&logo=opencv&logoColor=white)

## Project Goal

The primary objective of this project is to build and train a **Convolutional Neural Network (CNN)** capable of performing **age regression** from a dataset of facial images. The aim is to accurately predict the age of individuals, with a specific target of achieving a **Mean Absolute Error (MAE)** on the test set of **8 or less**.

## Business & Technical Challenge

Age regression from images has significant applications in demographics, personalized marketing, and security. However, it presents unique technical challenges:

* **High Feature Variability:** Facial aging patterns vary widely across ethnicities and individuals.
* **Architecture Selection:** Deep networks like **ResNet50** are powerful but require careful tuning to avoid extreme overfitting on regression tasks.
* **Loss Function Optimization:** Balancing training speed (using MSE) with the final evaluation metric (MAE).

## 🛠️ Methodology and Architecture

I developed a deep learning pipeline using the following functions to ensure modularity and scalability:

1.  **`load_data(path, subset)`**: Efficiently loads and preprocesses the training and test datasets.
2.  **`create_model(input_shape)`**: Defines the architecture leveraging **ResNet50** as a backbone for feature extraction, followed by dense layers for regression.
3.  **`train_model(...)`**: Manages the training process over 3 epochs.

### Model Strategy:
* **Backbone:** ResNet50 (Pre-trained weights).
* **Optimizer:** Adam.
* **Loss Function:** Mean Squared Error (MSE) for faster convergence.
* **Evaluation Metric:** Mean Absolute Error (MAE).

## 📂 Dataset Description

The dataset consists of **7,591 facial images** with associated age labels. 

> **Note:** To maintain repository efficiency, only a **representative sample** of images is included in this repository. The full dataset used for training is sourced from [Insert Source/Kaggle Link if available].

## 🚀 Key Results

* **Final MAE on Test Set:** [Inserta aquí tu resultado, ej. 6.8]
* **Benchmark:** Surpassed the project goal of MAE < 8. 
* **Insight:** Even with significant overfitting in the training phase, the model maintained high generalization capability on the validation set, confirming the effectiveness of the ResNet50 architecture for this task.

## Future Enhancements

* Implement **Data Augmentation** (horizontal flips, rotation) to improve generalization.
* Experiment with **InceptionV3** or **EfficientNet** for comparison.
* Fine-tune the deeper layers of ResNet50 specifically for the facial feature dataset.

---

### 👤 Contact
If you're interested in Deep Learning implementations or have questions about the model, let's connect!

* **LinkedIn:** [www.linkedin.com/in/fernando-garza-trevino]
* **Email:** ferngarzau@gmail.com
