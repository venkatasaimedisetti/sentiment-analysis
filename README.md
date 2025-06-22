# Sentiment Analysis on Amazon Fine Food Reviews

This project builds a sentiment analysis web application using machine learning to classify Amazon food reviews as positive, neutral, or negative. It includes data preprocessing, model training, and deployment via Streamlit.

**[Click here to try the live app on Streamlit](https://sentiment-analysis-wev6fvvvxzqovu7pk7doz3.streamlit.app/)**

---

## Project Objectives

- Predict sentiment from textual food reviews with 3-class classification (positive, neutral, negative)
- Build and train an optimized machine learning model
- Create an interactive frontend using Streamlit
- Deploy the model and app to GitHub and Streamlit Cloud

---

##  Tools & Technologies Used

- **Python** (3.11)
- **Pandas**, **NumPy**, **scikit-learn** for data processing & modeling
- **TfidfVectorizer** for text vectorization
- **Logistic Regression** for classification
- **Streamlit** for UI and deployment
- **Git/GitHub** for version control

##  Folder Structure

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
