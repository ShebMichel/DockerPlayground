import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

def generate_synthetic_data(n_samples=500, seed=42):
    """
    Generate synthetic linear data with some noise.
    
    Parameters:
        n_samples (int): Number of data points to generate.
        seed (int): Random seed for reproducibility.
    
    Returns:
        X (np.ndarray): Feature array of shape (n_samples, 1)
        y (np.ndarray): Target array of shape (n_samples,)
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 100, size=(n_samples, 1))  # random values between 0 and 100
    noise = np.random.normal(0, 5, size=n_samples)      # Gaussian noise
    y = 2 * X.flatten() + noise                          # Linear relation with noise
    return X, y

def train_and_save_model(X, y, filename='model.pkl'):
    """
    Train a linear regression model and save it to disk.
    
    Parameters:
        X (np.ndarray): Feature array
        y (np.ndarray): Target array
        filename (str): Path to save the model
    """
    try:
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, filename)
        print(f"Model trained and saved to '{filename}'!")
    except Exception as e:
        print(f"Error during training or saving the model: {e}")

if __name__ == "__main__":
    X, y = generate_synthetic_data(n_samples=500)
    train_and_save_model(X, y)


