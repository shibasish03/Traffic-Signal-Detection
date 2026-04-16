Traffic Sign Classification using Deep Learning
📌 Overview

This project implements a Traffic Sign Classification System using a Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model can classify traffic sign images into 43 different categories with high accuracy.

The system also includes a Streamlit web application that allows users to upload an image and get real-time predictions.

🎯 Features
Multi-class classification (43 traffic sign classes)
Custom CNN model built using PyTorch
Image preprocessing and transformation
Training with loss tracking
Test accuracy up to ~95%
Streamlit web interface for easy interaction
🧠 Technologies Used
Python
PyTorch
Torchvision
Pandas
OpenCV
Streamlit
📂 Dataset
Dataset: GTSRB (German Traffic Sign Recognition Benchmark) https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Source: Kaggle
Contains:
Training images (organized by class)
Test images with CSV labels
43 traffic sign categories
⚙️ Installation
git clone https://github.com/shibasish03/Traffic-Sign-Detection.git
cd traffic-sign-classification
pip install -r requirements.txt
▶️ Usage
🔹 Train the Model
python train.py
🔹 Run Streamlit App
streamlit run app.py
📊 Results
Training Loss decreased from 1.29 → 0.04
Test Accuracy: ~95%
Model shows strong classification performance across multiple classes
🖥️ Output
Predicts traffic sign class from uploaded image
Displays result in real-time using Streamlit
📸 Screenshots

(Add your screenshots here)

Training graphs
Confusion matrix
Streamlit UI
🚀 Future Improvements
Use advanced models (ResNet, EfficientNet)
Add data augmentation
Improve real-time detection with OpenCV
Deploy on mobile or edge devices
📄 License

This project is for educational purposes.

👨‍💻 Author

Shibasish Bhattacharjee
Artificial Intelligence InternsElite (March 2026
