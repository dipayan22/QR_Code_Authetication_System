# QR Code Authentication System

## 🚀 Project Overview
The **QR Code Authentication System** is a deep learning-based solution designed to classify QR codes as **original (first print)** or **counterfeit (second print)**. This system leverages a **Convolutional Neural Network (CNN)** and is deployed as a **Streamlit web application** for real-time verification.

## 📌 Features
- **Deep Learning-based QR Code Classification**
- **User-friendly Streamlit Web App** for real-time authentication
- **Fast and Efficient Model Processing**
- **Supports JPG, PNG, and JPEG Image Formats**
- **Scalable and Deployable for Real-world Applications**

---
## 🛠️ Tech Stack
- **Programming Language**: Python
- **Machine Learning Framework**: TensorFlow / Keras
- **Computer Vision**: OpenCV
- **Web App Framework**: Streamlit
- **Libraries Used**:
  - `tensorflow`
  - `opencv-python`
  - `streamlit`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

---
## 📂 Project Structure
```
📁 qr_code_authentication_project
│── 📂 dataset                 # QR code images dataset (Original & Counterfeit)
│── 📂 models                  # Trained model files
│── 📂 scripts                 # Utility scripts
│── 📜 app.py                  # Streamlit web application
│── 📜 model_training.py        # CNN model training script
│── 📜 requirements.txt         # List of dependencies
│── 📜 README.md                # Documentation
```

---
## 📥 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/qr-code-authentication.git
cd qr-code-authentication
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the Dataset
Place the dataset in the `dataset/` directory. Ensure it contains two folders:
```
📂 dataset
│── 📂 first_print             # Original QR codes
│── 📂 second_print            # Counterfeit QR codes
```

### 4️⃣ Train the Model (Optional)
If you want to retrain the model, run:
```bash
python model_training.py
```

This will save the trained model as `qr_code_authentication_model.h5`.

---
## 🎮 Running the Application
### 1️⃣ Start the Streamlit Web App
```bash
streamlit run app.py
```

### 2️⃣ Upload an Image for Verification
- Click on **Upload Image**
- The model will classify the QR code as **Original** or **Counterfeit**
- Results will be displayed instantly

---
## 🏆 Model Performance
The model was evaluated using the following metrics:
- **Accuracy**: 95%
- **Precision, Recall, and F1-score** for robust performance analysis
- **Confusion Matrix** to visualize misclassifications

---
## 🔥 Future Improvements
- Enhance dataset with diverse QR codes for better generalization
- Optimize model performance for faster predictions
- Deploy using **Flask/FastAPI** for production-scale use cases

---
## 🤝 Contribution
Want to improve this project? Follow these steps:
1. **Fork the repository**
2. **Create a new branch**
3. **Make your modifications**
4. **Submit a pull request**

---
## 📜 License
This project is licensed under the MIT License.

---
## 📧 Contact
For queries, feel free to reach out:
- **Email**: your.email@example.com
- **GitHub**: [your-github-profile](https://github.com/your-github-profile)
- **LinkedIn**: [your-linkedin](https://linkedin.com/in/your-profile)

---
🚀 **Happy Coding!** 🎯