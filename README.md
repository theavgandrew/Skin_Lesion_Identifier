# 🩺 Skin Lesion Classifier using HAM10000 and PyTorch

A computer vision model trained to classify dermatoscopic images of skin lesions into seven diagnostic categories using the HAM10000 dataset. This project explores the use of deep learning in dermatology and implements best practices in interpretability, class imbalance handling, and web-based deployment for accessibility.



## 📌 Project Summary

This project uses transfer learning with a pretrained ResNet18 architecture, fine-tuned on the HAM10000 dataset, to classify dermatoscopic images into 7 skin lesion types. The pipeline includes data preprocessing, augmentation, model training, performance evaluation, Grad-CAM visualization for interpretability, and deployment via a Streamlit web app.



## 📁 Dataset

**Source:** [Kaggle – HAM10000 ("Human Against Machine with 10000 training images")](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Classes:**
- `mel` – Melanoma  
- `nv` – Melanocytic nevi  
- `bkl` – Benign keratosis-like lesions  
- `bcc` – Basal cell carcinoma  
- `akiec` – Actinic keratoses  
- `vasc` – Vascular lesions  
- `df` – Dermatofibroma

**Metadata Includes:**
- Image ID  
- Diagnosis  
- Age  
- Sex  
- Anatomical Site  



## 🚀 Features

- ✅ Transfer Learning with ResNet18  
- ✅ Weighted loss for class imbalance  
- ✅ Data augmentation and normalization  
- ✅ Precision, Recall, F1-score evaluation per class  
- ✅ Grad-CAM heatmaps for interpretability  
- ✅ Streamlit app for live predictions + visualizations  
- ✅ Deployment-ready code



## 🔧 Workflow Overview

1. **Data Preparation**
   - Images resized to 224×224
   - Normalization with ImageNet stats
   - Augmentation: Random horizontal flips, rotations
   - Stratified train/validation split
   - Custom PyTorch `Dataset` and `DataLoader`

2. **Model Training**
   - ResNet18 pretrained on ImageNet
   - Final layer replaced with 7-class output
   - `CrossEntropyLoss` with class weights
   - `Adam` optimizer with learning rate scheduling
   - Accuracy, loss, and F1-score logged per epoch

3. **Evaluation**
   - Confusion matrix + classification report
   - Per-class F1, precision, and recall
   - Visualizations of correct vs incorrect predictions

4. **Interpretability**
   - Grad-CAM overlays show where the model focuses
   - Comparison between raw image and attention heatmap

5. **Deployment**
   - Streamlit app with:
     - Image uploader
     - Prediction + confidence score
     - Grad-CAM visualization
   - Deployed via Hugging Face Spaces



## 📊 Sample Results

| Metric | Score |
|--------|-------|
| Accuracy | 85.2% |
| Weighted F1 Score | 84.7% |
| Best Class | `nv` – 94% recall |
| Hardest Class | `akiec` – 59% recall |

<p align="center">
  <img src="assets/sample_gradcam.png" width="500">
  <br><em>Grad-CAM heatmap showing model focus</em>
</p>



## 💡 Lessons Learned

- Handling real-world data imbalance is essential for healthcare models  
- Interpretability (Grad-CAM) builds trust in AI predictions  
- Streamlit is a fast and effective way to make models accessible  
- Transfer learning dramatically accelerates development time



## 🛠 Future Work

- Incorporate metadata (age, sex, site) into a multi-input model  
- Experiment with other architectures (DenseNet, EfficientNet)  
- Add test-time augmentation  
- Deploy as a mobile/web hybrid app with TensorFlow Lite  

---

## 📎 Run the Project

```bash
# Setup environment
pip install -r requirements.txt

# Train model
python train.py

# Evaluate and visualize
python evaluate.py

# Run Streamlit app
streamlit run app.py
