# 🩺 Skin Lesion Classifier (HAM10000 + PyTorch)

A deep learning-based classifier built to identify skin lesions from dermatoscopic images using the HAM10000 dataset. This project was developed to explore the application of AI in healthcare, particularly dermatology, and to strengthen my foundations in computer vision using PyTorch.



## 📁 Dataset

This project uses the **[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)** ("Human Against Machine with 10000 training images") dataset, which consists of 10,015 dermatoscopic images labeled with one of the following 7 skin lesion categories:

- `mel` - Melanoma  
- `nv` - Melanocytic nevi  
- `bkl` - Benign keratosis-like lesions  
- `bcc` - Basal cell carcinoma  
- `akiec` - Actinic keratoses  
- `vasc` - Vascular lesions  
- `df` - Dermatofibroma



## 🧠 Project Goals

- Classify dermatoscopic images into 7 lesion types
- Use a convolutional neural network architecture built with PyTorch
- Handle class imbalance and optimize performance using basic techniques
- Visualize predictions and model confidence
- Evaluate performance using metrics beyond accuracy (e.g. confusion matrix)



## 🚀 Workflow

1. **Data Preprocessing**  
   - Image resizing to 224×224  
   - Data augmentation (random flips, rotation, normalization)  
   - CSV metadata parsing for class labels

2. **Model**  
   - CNN-based architecture using PyTorch  
   - Fine-tuned on the HAM10000 dataset

3. **Training & Evaluation**  
   - Loss/accuracy tracking  
   - Validation performance monitoring  
   - Confusion matrix analysis and misclassification visualization



## 📊 Results

*To be filled in once training completes*



## 🛠 Future Improvements

- Add support for Grad-CAM to visualize model attention  
- Compare custom CNN vs. transfer learning  
- Deploy as a web app using Streamlit or Flask  



## 📌 Motivation

Skin cancer is one of the most common cancers globally. Early detection can significantly improve prognosis, yet access to trained dermatologists remains limited in many areas. This project is a step toward understanding how deep learning can assist clinicians in diagnostic support.
