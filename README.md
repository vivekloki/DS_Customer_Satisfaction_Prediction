# 😊 Customer Satisfaction Prediction (Classification)

This project predicts customer satisfaction using machine learning models. The application is built with **Streamlit** for an interactive user experience.

---

## 📌 Features
✅ Load and preprocess customer satisfaction dataset  
✅ Train multiple classification models  
✅ Automatically selects the best model based on performance metrics  
✅ Saves the best model using `joblib`  
✅ Streamlit-based UI for customer satisfaction prediction  

---

## 🛠️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/Customer-Satisfaction-Prediction.git
cd Customer-Satisfaction-Prediction
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Preprocess Data
bash
Copy
Edit
python scripts/data_process.py
5️⃣ Train Models
bash
Copy
Edit
python scripts/train_models.py
6️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
📂 Project Structure
bash
Copy
Edit
📂 Customer-Satisfaction-Prediction
│── 📜 app.py                  # Streamlit app
│── 📜 requirements.txt         # Required dependencies
│── 📂 scripts
│   ├── 📜 data_process.py      # Data preprocessing script
│   ├── 📜 train_models.py      # Model training script
│── 📜 best_model.pkl           # Trained model
│── 📜 README.md                # Project documentation
