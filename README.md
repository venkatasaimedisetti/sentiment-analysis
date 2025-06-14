# 🧠 Sentiment Analysis on Amazon Fine Food Reviews

This project builds a sentiment analysis web application using machine learning to classify Amazon food reviews as **positive**, **neutral**, or **negative**. It includes data preprocessing, model training, and deployment via Streamlit.

---

## 📌 Project Objectives

- Predict sentiment from textual food reviews with 3-class classification (positive, neutral, negative)
- Build and train an optimized machine learning model
- Create an interactive frontend using Streamlit
- Deploy the model and app to GitHub and Streamlit Cloud

---

## ⚙️ Tools & Technologies Used

- **Python** (3.11)
- **Pandas**, **NumPy**, **scikit-learn** for data processing & modeling
- **TfidfVectorizer** for text vectorization
- **Logistic Regression** for classification
- **Streamlit** for UI and deployment
- **Git/GitHub** for version control

## 🗂️ Folder Structure

sentiment-analysis/
│
├── data/ # Contains Reviews.csv (not pushed to GitHub)
├── models/ # Pickled model and vectorizer
├── src/
│ ├── train_model.py # Script to train and save the model
│ └── app.py # Streamlit app for prediction
├── .gitignore # Ignored files and folders
├── requirements.txt # Python dependencies
└── README.md # Project overview

yaml
Copy
Edit

---

## 🚀 How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
### 2. Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate    # On Windows
### 3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
### 4. Train the Model (if not already trained)
bash
Copy
Edit
python src/train_model.py
### 5. Run the Streamlit App
bash
Copy
Edit
streamlit run src/app.py
🌐 Deployment
The app can be deployed using Streamlit Cloud

Ensure app.py, requirements.txt, and model files are available in the repository root or configured correctly

For large files, use .gitignore and avoid uploading data files > 100 MB to GitHub

🔒 .gitignore Highlights
kotlin
Copy
Edit
venv/
__pycache__/
*.pkl
data/
models/
.DS_Store
