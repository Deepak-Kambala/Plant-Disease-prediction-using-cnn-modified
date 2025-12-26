training_model.h5 
link:- https://drive.google.com/file/d/1OfAiw-D6W5aTEVcaTqytJuf7oz-vgYvx/view?usp=sharing

traning_model.h5 should be in same folder

# 🌿 Plant Disease Prediction (Accuracy 95%)

A **Streamlit-based web app** that uses a deep learning model to **detect plant diseases from leaf images** with **95% accuracy**.  
Upload a leaf photo, and the app instantly identifies the disease and suggests treatment options—helping farmers and gardeners protect crops and reduce losses.

---

## ✨ Features
- 🖼 **AI-Powered Detection** – Upload a plant leaf image and get real-time disease predictions.  
- 📚 **Pre-Trained Model** – Achieves **95% accuracy** using a convolutional neural network.  
- 🌱 **User-Friendly Interface** – Built with [Streamlit](https://streamlit.io) for a fast, interactive experience.  
- 📊 **Result Insights** – Displays prediction confidence and recommended actions.  

---

## 🛠 Tech Stack
- **Frontend/Backend:** [Streamlit](https://streamlit.io/)  
- **Machine Learning:** TensorFlow / Keras (CNN model)  
- **Language:** Python 3.x  

---

## 📂 Project Structure
model1_Accuracy95/
│
├── offline.py # Main Streamlit app
├── offline_translations.py
├── requirements.txt # Dependencies
├── home_page.jpeg
└── README.md # Documentation
|__ training_model.h5

 🚀 Getting Started

### 1️⃣ Clone the Repository
bash
git clone https://github.com/MohdAbdulRah/NA09_CodeStorm.git
cd PlantDiseasePredicton/model1_Accuracy95

### 2️⃣ Create & Activate Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the Streamlit App
streamlit run offline.py

### 5️⃣ Open in Browser

Streamlit will display a local URL (e.g., http://localhost:8501). Open it to use the app.

### 📸 Usage

Click Browse files to upload a clear photo of a plant leaf.

Wait for the AI model to analyze the image.

View the predicted disease, confidence score, and treatment suggestions.

### 📊 Model Details

Architecture: Convolutional Neural Network (CNN)

Accuracy: 95% on test dataset

Trained using a publicly available plant disease dataset.

### 🌟 Future Improvements

Expand the model to cover more plant species.

Add multi-language support for farmers worldwide.

Integrate weather and soil data for advanced predictions.

### 🤝 Contributing

Contributions are welcome! Fork the repo, create a branch, and submit a pull request.
