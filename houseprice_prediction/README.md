# 📌 House Price Prediction – Interactive ML Training & Deployment with CSV Upload

An **end-to-end machine learning project** featuring **dynamic CSV upload, on-demand model training, real-time predictions, and interactive visualizations** using **Streamlit** inside a **Docker container**. Designed for **intermediate ML engineers and data scientists** to showcase modern ML workflow and deployment skills with user-driven data input.

![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![ML](https://img.shields.io/badge/ML-Interactive%20Training-red)

---

## 🆕 What's New in This Version?

### Revolutionary Features:
- **🎯 CSV Upload Interface** – No more hardcoded datasets! Users upload their own data
- **🔧 Dynamic Feature Selection** – Choose any columns as features or target
- **⚡ On-Demand Training** – Train models in real-time with a single click
- **💾 Session-Based Storage** – No file system dependencies, everything in memory
- **📊 Interactive Configuration** – Fully customizable without code changes
- **🎨 Enhanced Visualizations** – Beautiful charts with detailed statistics

---

## Application Preview

![ML Prediction App](assets/images/house_predict.png)

---

## 🎯 Problem Statement

Predict **house prices** (or any numerical target) based on user-provided features such as:
- House area (sq ft)  
- Number of rooms  
- House age (years)  

**But now you can use ANY dataset!** This flexible approach supports:
- Loan approval prediction  
- Customer churn prediction  
- Energy consumption forecasting  
- Sales forecasting
- Stock price prediction
- Medical cost estimation
- Or any regression problem!

---

## 🧠 Tech Stack

- **Python 3.10**  
- **scikit-learn** (Linear Regression with dynamic training)  
- **pandas / numpy** (data processing)  
- **matplotlib** (visualizations)  
- **Streamlit** (interactive UI with file upload)  
- **Docker** (containerization)

---

## 📁 Project Structure

```
houseprice_prediction/
├── app.py                      # Complete Streamlit application (all-in-one)
├── Dockerfile                  # Simplified Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── sample_housing_data.csv    # Sample dataset for testing
```

### 🎉 Simplified Architecture
**Before**: Complex structure with multiple folders (`src/`, `models/`, `data/`) and scripts  
**Now**: Single `app.py` file with everything integrated – training, prediction, and visualization!

---

## ⚡ Prerequisites

- **Docker** installed on your machine  
- **Any CSV dataset** with numerical features and a target column
- Optional: **Python 3.10+** for local testing

### Local Testing (Optional)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Quick Start Guide

### Step 1: Build Docker Image
```bash
docker build -t ml-houseprice-app .
```

### Step 2: Run Docker Container
```bash
docker run -p 8501:8501 ml-houseprice-app
```

### Step 3: Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://0.0.0.0:8501
```

---

## 📊 How to Use the Application

### 1️⃣ Upload Your Dataset
- Click **"Browse files"** button
- Select your CSV file (must contain numerical data)
- See instant data preview and statistics

### 2️⃣ Configure Features
- **Select Feature Columns**: Choose one or more columns as input features
- **Select Target Column**: Choose the column you want to predict
- The app validates your selection automatically

### 3️⃣ Train the Model
- Click **"🚀 Train Model"** button
- Watch the training progress
- See success confirmation with 🎈 celebration!

### 4️⃣ View Model Performance
- Check the **sidebar** for:
  - **RMSE** (Root Mean Squared Error)
  - **R² Score** (coefficient of determination)

### 5️⃣ Make Predictions
- Enter values for each feature in the input fields
- Click **"Predict Price"** button
- Get instant prediction result

### 6️⃣ Analyze Results
- View **Actual vs Predicted** scatter plot
- See prediction statistics:
  - Mean Actual Price
  - Mean Predicted Price
  - Mean Absolute Error

---

## 📝 CSV Format Requirements

Your CSV file should:
- Have **headers** in the first row
- Contain **numerical values** for features and target
- Have **no missing values** in selected columns (or handle them beforehand)

### Example CSV Structure:
```csv
area,rooms,age,price
1500,3,10,250000
2000,4,5,350000
1200,2,15,200000
1800,3,8,280000
```

### Using the Sample Dataset
A `sample_housing_data.csv` is included for testing:
```bash
# Just upload this file in the app to see it in action!
```

---

## 💻 Application Features

### 🎯 Core Features
- **📁 CSV File Upload** – Drag & drop or browse to upload
- **👀 Data Preview** – Instant view of your data
- **📊 Data Statistics** – Rows, columns, missing values count
- **🎚️ Dynamic Feature Selection** – Choose features/target interactively
- **🚀 One-Click Training** – Train models instantly
- **💰 Real-Time Predictions** – Get predictions in milliseconds
- **📈 Beautiful Visualizations** – Professional scatter plots with perfect prediction line
- **📊 Model Metrics** – RMSE and R² displayed prominently
- **🔄 Session Management** – Upload different datasets without restarting

### 🎨 UI/UX Highlights
- Clean, modern interface
- Responsive column layouts
- Color-coded metrics
- Interactive number inputs
- Progress indicators
- Success/error notifications
- Celebration animations on successful training

---

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Linear Regression (scikit-learn)
- **Training/Test Split**: 80/20 ratio
- **Random State**: 42 (reproducible results)

### Data Processing
- Dynamic feature extraction based on user selection
- Automatic NumPy array conversion
- Real-time validation of column selections

### Session State Management
The app uses Streamlit's session state to store:
- Trained model object
- Model metrics (RMSE, R²)
- Feature column names
- Test predictions for visualization

### Why Session State?
- ✅ No file system dependencies
- ✅ Faster access to model
- ✅ Works seamlessly in Docker
- ✅ Easy to reset (just refresh page)
- ⚠️ Note: Data is lost on page refresh (by design for security)

---

## 🐳 Docker Configuration

### Dockerfile Highlights
```dockerfile
FROM python:3.10-slim          # Lightweight base image
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy dependencies first (layer caching)
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .                  # Copy application
EXPOSE 8501                    # Streamlit default port
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Key Improvements
- **No pre-training required** – Model trains on-demand
- **Single file deployment** – Just `app.py`
- **Faster builds** – No data processing at build time
- **Smaller image size** – Removed unnecessary files

---

## 🎓 Use Cases & Extensions

### Current Implementation
- House price prediction
- Real estate valuation

### Easy Extensions (Same Code!)
- **Financial**: Loan default prediction, credit scoring
- **Healthcare**: Medical cost estimation, patient readmission
- **Retail**: Sales forecasting, demand prediction
- **Energy**: Consumption forecasting, load prediction
- **Manufacturing**: Quality prediction, defect rate estimation
- **Marketing**: Customer lifetime value, conversion rate

### How to Extend
Just upload a different CSV with:
- Your features → Select them in the app
- Your target → Select it in the app
- Click train → Done! 🎉

---

## 🔍 Troubleshooting

### Common Issues

**Problem**: Error when uploading CSV  
**Solution**: Ensure your CSV has proper headers and numerical values

**Problem**: Training fails  
**Solution**: Check that selected columns don't have missing/non-numeric values

**Problem**: Can't select target column  
**Solution**: Make sure target column is different from feature columns

**Problem**: Port 8501 already in use  
**Solution**: 
```bash
# Use a different port
docker run -p 8502:8501 ml-houseprice-app
# Then access at http://localhost:8502
```

**Problem**: Model performance is poor  
**Solution**: Try:
- Adding more features
- Using more training data
- Checking for outliers in your data
- Consider feature engineering

---

## 📈 Performance Tips

### For Better Predictions
1. **More data is better** – Upload larger datasets for better model performance
2. **Relevant features** – Choose features that logically affect your target
3. **Clean data** – Remove outliers and handle missing values beforehand
4. **Scale appropriately** – For better results, consider feature scaling

### For Faster Training
- Use datasets with < 100,000 rows for quick training
- Reduce number of features if training is slow
- For large datasets, consider training locally first

---

## 🔐 Security Considerations

- CSV files are processed in-memory only (not saved to disk)
- Session state clears on page refresh
- No persistent storage of user data
- Docker container runs with minimal permissions
- No external data transmission

---

## 🛣️ Roadmap & Future Enhancements

Potential additions:
- [ ] Support for more ML algorithms (Random Forest, XGBoost, etc.)
- [ ] Automatic feature engineering
- [ ] Model comparison tools
- [ ] Export trained models
- [ ] Feature importance visualization
- [ ] Cross-validation support
- [ ] Hyperparameter tuning interface
- [ ] Support for categorical features
- [ ] Time series forecasting mode
- [ ] Multi-model ensemble

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 📄 License

**MIT License** – free to use, modify, and distribute.

```
Copyright (c) 2026 Michel M. Nzikou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🌟 Why This Project Stands Out

### For Recruiters & Hiring Managers
- ✅ **Modern ML Workflow** – Shows understanding of end-to-end ML pipeline
- ✅ **Production-Ready** – Docker deployment, clean code, error handling
- ✅ **User-Centric Design** – Interactive UI, real-time feedback
- ✅ **Flexible Architecture** – Works with any regression dataset
- ✅ **Best Practices** – Session management, validation, documentation

### For Data Scientists & ML Engineers
- ✅ **Quick Prototyping** – Test models on different datasets instantly
- ✅ **Educational Tool** – Learn model training and deployment
- ✅ **Portfolio Project** – Showcase deployment skills
- ✅ **Extensible Framework** – Easy to add new features
- ✅ **Real-World Application** – Solves actual prediction problems

---

## 📞 Contact & Support

**Author**: Michel M. Nzikou  
**Date**: February 2026  
**Organization**: DMN SOLUTIONS  
**Location**: Perth, WA, Australia

For questions, suggestions, or collaboration:
- GitHub: [ShebMichel](https://github.com/ShebMichel)
- Project Repository: [DockerPlayground/houseprice_prediction](https://github.com/ShebMichel/DockerPlayground/tree/main/houseprice_prediction)

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) – Amazing framework for ML apps
- [scikit-learn](https://scikit-learn.org/) – Powerful ML library
- [Docker](https://www.docker.com/) – Containerization platform

Inspired by the need for flexible, user-friendly ML deployment solutions.

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

---

<div align="center">

### ⭐ If you find this project useful, please give it a star! ⭐

**Happy Predicting! 🚀**

</div>
