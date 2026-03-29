# 🧠 Stress Level Detection
## 📌 Overview
This project is a Machine Learning-based system that detects human stress levels using computer vision and deep learning techniques. It can process images or real-time webcam input to predict stress levels.
---
## 🚀 Features
* Real-time stress detection (webcam)
* Image-based prediction
* Custom-trained ML model
* Modular pipeline (training + inference)
---
## 🛠️ Tech Stack
* Python
* OpenCV
* PyTorch / TensorFlow
* NumPy / Pandas
---
## 📂 Dataset
Due to GitHub file size limitations, the dataset is not included in this repository.
👉 Download dataset from here:
**[https://drive.google.com/file/d/1DEtp_ZPYn5IVRgMzUQtKfCzLYADilD_4/view?usp=drive_link]**
After downloading, place the dataset in the project root directory:
```bash
/project-folder/data/
```
---

## ⚙️ Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/shuhb021/Stress-level-Detection-.git
cd Stress-level-Detection-
```
### 2. Create Virtual Environment
```bash
python -m venv venv
```
Activate:
```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```
---
### 3. Install Dependencies
```bash
pip install -r requirement.txt
---
### 4. Run the Project
#### ▶️ Webcam Detection
```bash
python webcam_detect.py
```
#### ▶️ Image Detection
```bash
python detect_image.py
```
---
## ⚠️ Files Not Included in Repo
The following files are intentionally excluded:
* `data211.csv` → Dataset file (>100MB)
* `*.pth` → Trained model weights
* `venv/` → Virtual environment
👉 Reason
* GitHub file size limit (100MB)
* Keeps repository clean and lightweight
---
## 📥 How to Use Missing File 
* Download dataset from the link above
* Train the model using:
```bash
python train.py
* OR use your own dataset
---
## 📈 Future Improvements
* Web deployment (Streamlit / Flask)
* Model optimization
* UI enhancement
---
## 🤝 Contribution
Feel free to fork this repo and improve the project.

## 📬 Contact
For any queries or collaboration, connect with me on LinkedIn.
---

⭐ If you like this project, don't forget to star the repo!
