# 📌 End-to-End ML Model Deployment with Streamlit & Docker 

An end-to-end machine learning project that covers data preprocessing, model training, evaluation, and deployment using Streamlit inside a Docker container. Designed for intermediate ML engineers and data scientists.

![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)

🎯 Problem Statement

Predict house prices based on numerical features (classic, understandable, and recruiter-friendly).

You can later swap this with:

- Loan approval prediction
- Customer churn
- Energy consumption forecasting

---

## Application Preview

![ML Prediction App](assets/images/house_predict.png)

🧠 Tech Stack

Python 3.10

scikit-learn

pandas / numpy

joblib

Streamlit

Docker

## Project Structure
```
├── app
│   └── app.py
├── data
│   ├── create_synthetic_data.py
│   └── housing.csv
├── Dockerfile
├── models
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── predict.py
    ├── preprocess.py
    ├── train.py
    └── utils.py

```
---

## Prerequisites

- Docker installed on your machine  
- Python 3.11 (for local testing if needed)  
- Create a new environment name:petenv

---

## Step 1: Use Preexisting Model

We leverage the pretrained ImageNet classes to enable predictions on any uploaded image. The class labels are fetched directly from the PyTorch GitHub repository:

```bash
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
```
A simple API request is used to load the class names, which are then used to interpret the model’s predictions.

---

## Step 2: Run the Flask App in Docker

1. Build the Docker image:

```bash
docker build -t ml-houseprice_prediction-app .
```

2. Run the Docker container:

```bash
docker run -p 8501:8501 ml-houseprice_prediction-app
```

3. Successful startup output will look like:

```
* Running on http://127.0.0.1:8501
* Running on http://172.17.0.2:8501
```

---

## Step 3: Test the App

Open your browser and navigate to:

http://127.0.0.1:8501

You can now enter numeric values into the web interface and see predictions from the trained linear regression model instantly.

---

## HTML Interface

The application features a simple HTML front-end that lets users:

Select the size of your house, the number of units and the years, 

Click “Predict” to generate the predicted price of the house.

---

## Notes

- This project uses streamlit development server. 

---

## License

MIT License – free to use, modify, and distribute.

## References:
  - Michel M. Nzikou, Jan 2026 @ DMN SOLUTIONS, Perth, WA, Australia 
