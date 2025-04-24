# Lung_Cancer_Detection
# �� Pneumonia Detection App via Chest X-ray and Symptoms

A deep learning-based application for early pneumonia screening from X-rays and symptoms.

### 🛠️ Technologies
- Python 3.8.20
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Tkinter
- PIL
- Scikit-learn
- Matplotlib

## 📌 Features

- 🖼 Load and analyze chest X-ray images (grayscale)
- 🧠 Deep learning prediction using a trained CNN model
- 📝 Input form for symptoms (e.g., cough, chest pain, smoking, etc.)
- 📊 Provides predictive result: low or high risk of lung cancer
- 🔄 Clean, simple, and intuitive graphical interface

---

### 📦 Installation
```bash
git clone https://github.com/Tommy240302/Lung_Cancer_Detection
cd https://github.com/Tommy240302/Lung_Cancer_Detection
pip install -r requirements.txt
```

### ▶️ Run
```bash
python app.py
```

### 🧪 Dataset
- Kaggle Chest X-ray Pneumonia Dataset
- https://www.kaggle.com/code/bonicajalakam/lung-cancer-detection-using-cnn/input
- Symptoms: [Custom structured medically](https://www.kaggle.com/datasets/humairmunir/lung-cancer-risk-dataset/data)

### 📊 Results
- Accuracy: ~93%
- Precision/Recall: 99%
- Regularization: Dropout, Augmentation, Balanced Classes

### 📌 Future Work
- Database connection
- Detect more lung diseases
- Hospital software integration

## 🚀 Usage

```bash
python app.py
```

Once the GUI launches, you can:

- 📷 Load an X-ray image via **"Ảnh chụp X-Quang" (Load X-ray Image)**
- 🔍 Analyze the image via **"Kiểm tra ảnh" (Analyze Image)**
- 🔄 Clear image and result **"Làm mới" (Clear)**
- 📝 Enter symptoms via **"Nhập triệu chứng" (Enter Symptoms)**
- ✅ Get a prediction on potential lung cancer risk

---

## 🧪 Demo

### 🖥 Main Interface:

![detect_by_X-Quang](https://github.com/user-attachments/assets/84e7b3c6-894e-44cf-b41e-7098a0c206bb)

### 📷 Detect By X-Quang Image:

![image](https://github.com/user-attachments/assets/2131b2aa-45c5-4297-b176-05aa27e34772)

![image](https://github.com/user-attachments/assets/c3996abd-ebc6-4d46-aba1-744616361724)

### 📝 Symptom Input Form:

![detect_by_symptom](https://github.com/user-attachments/assets/4d783caa-8f58-4aa6-a83f-ebe1e9336eba)

---
### 📄 License
- This project is not licensed under the MIT License.


