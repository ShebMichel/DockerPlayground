import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title("🏠 House Price Prediction")

# -----------------------------
# CSV Upload Section
# -----------------------------
st.subheader("📁 Upload Training Data")

uploaded_file = st.file_uploader("Upload CSV file for training", type=['csv'])

if uploaded_file is not None:
    # Load the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    st.write("### Data Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Select features and target
    st.write("### Select Features and Target")
    
    columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        feature_columns = st.multiselect("Select Feature Columns",columns,default=columns[:-1] if len(columns) > 1 else []
        )
    # Handle categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_columns) > 0:
        st.info(f"📝 Categorical columns detected: {', '.join(categorical_columns)}")
        st.write("Converting categorical variables to numerical using Label Encoding...")
        
        # Store encoders in session state for prediction
        if 'label_encoders' not in st.session_state:
            st.session_state['label_encoders'] = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            st.session_state['label_encoders'][col] = le

    with col2:
        target_column = st.selectbox(
            "Select Target Column",
            columns,
            index=len(columns)-1 if len(columns) > 0 else 0
        )
    
    if st.button("🚀 Train Model"):
        if len(feature_columns) == 0:
            st.error("Please select at least one feature column!")
        elif target_column in feature_columns:
            st.error("Target column cannot be in feature columns!")
        else:
            with st.spinner("Training model..."):
                try:
                    # # Prepare data
                    # X = df[feature_columns].values
                    # y = df[target_column].values
                    
                    # # Split data
                    # X_train, X_test, y_train, y_test = train_test_split(
                    #     X, y, test_size=0.2, random_state=42
                    # )
                    # Prepare data
                    X = df[feature_columns].values
                    y = df[target_column].values

                    # Drop ID columns if they were selected as features
                    id_like_columns = ['id', 'ID', 'i_d', 'I_D', 'index', 'idx', 'Index', 'ID_', '_id']
                    id_cols_to_remove = [col for col in feature_columns if col.lower() in [x.lower() for x in id_like_columns]]

                    if id_cols_to_remove:
                        st.warning(f"⚠️ Removing ID-like columns from features: {', '.join(id_cols_to_remove)}")
                        feature_columns = [col for col in feature_columns if col not in id_cols_to_remove]
                        X = df[feature_columns].values
                        
                        # Update session state with cleaned features
                        st.session_state['feature_columns'] = feature_columns

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    # Scale the features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Try multiple models and pick the best
                    models_to_try = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, max_depth=10, learning_rate=0.1),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
                    }

                    best_model = None
                    best_model_name = None
                    best_r2 = -float('inf')
                    results = {}

                    for model_name, model in models_to_try.items():
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        results[model_name] = {'r2': r2, 'rmse': rmse}
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = model
                            best_model_name = model_name
                            y_pred_best = y_pred

                    # Show comparison
                    st.write("### 🔬 Model Comparison")
                    comparison_df = pd.DataFrame(results).T
                    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['r2']).highlight_min(axis=0, subset=['rmse']))

                    st.info(f"✅ Selected Model: **{best_model_name}** (R²: {best_r2:.2f})")

                    # Use best model
                    rmse = results[best_model_name]['rmse']
                    r2 = best_r2

                    # Check if R² meets threshold
                    if r2 < 0.95:
                        st.warning(f"⚠️ Best model R² is {r2:.2f}, below 95% threshold. Consider improving your data.")

                    # Save to session state
                    st.session_state['model'] = best_model
                    st.session_state['model_name'] = best_model_name
                    st.session_state['scaler'] = scaler
                    st.session_state['metrics'] = {'rmse': rmse, 'r2': r2}
                    st.session_state['feature_columns'] = feature_columns  
                    st.session_state['predictions'] = pd.DataFrame({
                        'actual': y_test,
                        'predicted': y_pred_best
                    })
                    st.success("✅ Model trained successfully!")  
                    st.balloons() 
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

# -----------------------------
# Model Performance (only show if model exists)
# -----------------------------
if 'metrics' in st.session_state:
    st.sidebar.header("📊 Model Performance")
    metrics = st.session_state['metrics']
    st.sidebar.metric("RMSE", f"{metrics['rmse']:.2f}")
    st.sidebar.metric("R² Score", f"{metrics['r2']:.2f}")

# -----------------------------
# Prediction Section (only show if model exists)
# -----------------------------
#if 'model' in st.session_state:
if 'model' in st.session_state and 'feature_columns' in st.session_state:
    st.subheader("🔢 Make Predictions")
    
    model = st.session_state['model']
    feature_columns = st.session_state['feature_columns']
    
    # Create input fields dynamically based on features
    cols = st.columns(min(3, len(feature_columns)))
    input_features = []

    for idx, feature_name in enumerate(feature_columns):
        with cols[idx % 3]:
            # Check if this feature was categorical
            if feature_name in st.session_state.get('label_encoders', {}):
                le = st.session_state['label_encoders'][feature_name]
                options = le.classes_.tolist()
                selected = st.selectbox(
                    f"{feature_name}",
                    options,
                    key=f"input_{feature_name}"
                )
                # Transform selection to numerical
                input_features.append(le.transform([selected])[0])
            else:
                value = st.number_input(
                    f"{feature_name}",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"input_{feature_name}"
                )
                input_features.append(value)

    if st.button("Predict Price"):
        # Check if model meets accuracy threshold
        if st.session_state['metrics']['r2'] < 0.95:
            st.error(f"❌ Model accuracy ({st.session_state['metrics']['r2']:.2%}) is below 95%. Please retrain with better data or features.")
        else:
            try:
                features = np.array(input_features).reshape(1, -1)
                features_scaled = st.session_state['scaler'].transform(features)  # Scale the input!
                prediction = st.session_state['model'].predict(features_scaled)[0]
                st.success(f"💰 Estimated Price: ${prediction:,.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    # -----------------------------
    # Visualization Section
    # -----------------------------
    st.subheader("📈 Model Predictions vs Actual")
    if 'predictions' in st.session_state:
        df_pred = st.session_state['predictions']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_pred["actual"], df_pred["predicted"], alpha=0.6)
        ax.plot(
            [df_pred["actual"].min(), df_pred["actual"].max()],
            [df_pred["actual"].min(), df_pred["actual"].max()],
            'r--', linewidth=2, label='Perfect Prediction'
        )
        ax.set_xlabel("Actual Price", fontsize=12)
        ax.set_ylabel("Predicted Price", fontsize=12)
        ax.set_title("Actual vs Predicted Prices", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Show prediction statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Actual", f"${df_pred['actual'].mean():,.2f}")
        with col2:
            st.metric("Mean Predicted", f"${df_pred['predicted'].mean():,.2f}")
        with col3:
            error = abs(df_pred['actual'] - df_pred['predicted']).mean()
            st.metric("Mean Absolute Error", f"${error:,.2f}")

else:
    st.info("👆 Please upload a CSV file to train the model and start making predictions!")
