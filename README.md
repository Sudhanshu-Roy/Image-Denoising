# 🚗 Emergency Vehicle Image Denoising using Convolutional Autoencoders

## 📌 Overview

Image noise is a common challenge in computer vision systems and can arise due to sensor limitations, poor lighting conditions, transmission errors, or environmental disturbances. Excessive noise reduces image quality and can negatively impact downstream tasks such as object detection, classification, and recognition.

This project presents a Deep Learning-based image denoising system built using a Convolutional Autoencoder. The model learns to reconstruct clean emergency vehicle images from noisy inputs by learning meaningful latent representations of the image.

---

## 🎯 Objectives

* Remove noise from emergency vehicle images
* Learn compressed image representations using Autoencoders
* Reconstruct cleaner versions of noisy images
* Build an end-to-end Computer Vision pipeline
* Deploy the model using Streamlit

---

## 🧠 Model Architecture

The project uses a Convolutional Autoencoder consisting of:

### Encoder

* Conv2D (32 Filters)
* MaxPooling2D
* Conv2D (64 Filters)
* MaxPooling2D
* Conv2D (128 Filters)
* MaxPooling2D

### Decoder

* Conv2DTranspose (128 Filters)
* Conv2D (128 Filters)
* Conv2DTranspose (64 Filters)
* Conv2D (64 Filters)
* Conv2DTranspose (32 Filters)
* Conv2D (32 Filters)
* Conv2D Output Layer (Sigmoid Activation)

The encoder compresses the noisy image into a latent representation, while the decoder reconstructs a cleaner version of the image.

---

## 🔄 Project Workflow

```text
Noisy Image
      ↓
Image Preprocessing
      ↓
Convolutional Autoencoder
      ↓
Denoised Image
```

---

## 📂 Dataset

The dataset consists of paired images:

```text
Train/
├── Clean/
└── Noisy/

Test/
└── Noisy/
```

Each noisy image has a corresponding clean image used as the target during training.

Due to GitHub storage limitations, the dataset is not included in this repository.

Dataset download instructions can be found in:

```text
data/dataset.txt
```

---

## ⚙️ Preprocessing

The following preprocessing steps are applied:

* Read image using OpenCV
* Convert BGR → RGB
* Convert image to float32
* Normalize pixel values to the range [0,1]

```python
image = image / 255.0
```

The same preprocessing pipeline is applied during inference.

---

## 🖼️ Results

The trained Autoencoder successfully removes a significant amount of noise while preserving the overall structure of emergency vehicles.

### Example Result

<img width="794" height="394" alt="image" src="https://github.com/user-attachments/assets/33e49de0-0986-4104-befd-cfd280c529b6" />

and reference them here.

---

## 🌐 Streamlit Application

A Streamlit application was developed to provide an interactive interface for image denoising.

Features:

* Upload noisy image
* Run denoising inference
* Visualize noisy image
* Visualize denoised image

---

## 📁 Project Structure

```text
Emergency-Vehicle-Image-Denoising/

├── app.py
├── assets/
├── data/
│   └── dataset.txt
├── models/
│   └── weights_best.weights.h5
├── notebooks/
│   └── Image_Denoising_Autoencoder.ipynb
├── src/
│   ├── model.py
│   ├── preprocess.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Sudhanshu-Roy/Image-Denoising.git

cd Emergency-Vehicle-Image-Denoising
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Streamlit
* Deep Learning
* Computer Vision

---

## 🔮 Future Improvements

* U-Net based denoising architectures
* Residual Autoencoders
* Perceptual Loss Functions
* GAN-based Image Restoration
* Real-time image denoising

---

## 👨‍💻 Author

**Sudhanshu Roy**

AI & Data Science Enthusiast

Interests:

* Deep Learning
* Computer Vision
* Generative AI
* Data Science

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
