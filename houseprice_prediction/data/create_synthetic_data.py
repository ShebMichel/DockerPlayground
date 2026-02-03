import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
N = 1000

# Generate features
area = np.random.randint(500, 5000, size=N)         # sq ft
rooms = np.random.randint(1, 8, size=N)            # number of rooms
age = np.random.randint(0, 50, size=N)             # house age in years

# Generate target price (simple linear formula + noise)
price = area * 300 + rooms * 10000 - age * 500 + np.random.normal(0, 50000, N)

# Create DataFrame
df = pd.DataFrame({
    "area": area,
    "rooms": rooms,
    "age": age,
    "price": price
})

# Create data directory if not exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Save CSV
csv_path = data_dir / "housing.csv"
df.to_csv(csv_path, index=False)

print(f"Generated dataset at {csv_path}")
