FastAPI_SentimentAnalyzer

📝 Project Description
This project demonstrates a robust and scalable Sentiment Analysis API built with FastAPI, complemented by an interactive Streamlit web application for easy demonstration and interaction. The backend API leverages three powerful machine learning models—Logistic Regression, Naive Bayes, and Support Vector Machines (SVM)—to classify the sentiment of text inputs. The Streamlit front-end allows users to input text and immediately see the sentiment predictions from each model. This combined solution provides an efficient and easily deployable tool for integrating and showcasing sentiment analysis capabilities, ideal for tasks such as social media monitoring, customer feedback analysis, and content moderation.

✨ Features

Multiple Sentiment Models (FastAPI Backend): Utilizes Logistic Regression, Naive Bayes, and SVM for diverse sentiment classification capabilities.

FastAPI Framework: Built with FastAPI for high performance, automatic interactive API documentation (Swagger UI/ReDoc), and easy development.

Interactive Web UI (Streamlit Frontend): Provides a user-friendly interface to input text and view real-time sentiment predictions from the FastAPI backend.

Scalable & Asynchronous: Designed to handle multiple requests concurrently, ensuring responsiveness under load.

Docker Support: Containerized for easy deployment and consistent environments across different systems.

Clear API Endpoints: Intuitive and well-documented endpoints for seamless integration.

🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites

Python 3.8+

pip (Python package installer)

Docker (optional, for containerized deployment)

Installation

Clone the repository:

git clone https://github.com/RajatPawar33/FastAPI_SentimentAnalyzer.git

cd FastAPI_SentimentAnalyzer

Create a virtual environment (recommended):

python -m venv venv

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Running the Applications

This project has two main components: the FastAPI backend API and the Streamlit frontend.

1. Run the FastAPI Backend API

To run the API locally using Uvicorn:

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The FastAPI API will be available at http://localhost:8000. You can access its interactive documentation (Swagger UI) at http://localhost:8000/docs.

2. Run the Streamlit Frontend Application

In a separate terminal and after the FastAPI backend is running, launch the Streamlit app:

streamlit run app.py

The Streamlit application will typically open in your web browser at http://localhost:8501.

⚙️ API Endpoints

Once the FastAPI API is running, you can access the interactive API documentation (Swagger UI) at http://localhost:8000/docs or ReDoc at http://localhost:8000/redoc.

POST /predict_sentiment

Analyzes the sentiment of a given text using all implemented models.

URL: /predict_sentimen

Method: POST

Request Body:

JSON

{
  "text": "Your input text here."
}

Response:

JSON
{
  "text": "Your input text here.",
  
  "logistic_regression_sentiment": "Positive",
  
  "naive_bayes_sentiment": "Negative",
  
  "svm_sentiment": "Neutral"
}

(Note: Sentiment values will be "Positive", "Negative", or "Neutral" based on model predictions.)

GET /

Root endpoint, primarily for checking if the API is running.

URL: /

Method: GET

Response:

JSON
{
  "message": "Sentiment Analysis API with FastAPI"
}

🛠️ Technologies Used

FastAPI: Web framework for building high-performance APIs.

Streamlit: For creating interactive web applications and dashboards.

Uvicorn: ASGI server for running the FastAPI application.

Scikit-learn: For machine learning models (Logistic Regression, Naive Bayes, SVM).

NLTK: For text preprocessing (e.g., tokenization, stop word removal).

Pandas: For data handling.

Docker: For containerization.

Requests: For the Streamlit app to communicate with the FastAPI backend.

🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

Fork the repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.
