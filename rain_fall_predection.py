import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\admin\\Downloads\\bangalore-rainfall-data-1900-2024-sept.csv'
data = pd.read_csv(file_path)

# Data preprocessing
# Drop rows with missing values
cleaned_data = data.dropna()

# Features (monthly rainfall data) and target (Total annual rainfall)
X = cleaned_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']]
y = cleaned_data['Total']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Predict rainfall for a given date
def predict_rainfall_for_date(date_str):
    """Predict total rainfall for a given date."""
    # Parse the date to determine the month
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.strftime('%b')  # Get month abbreviation (e.g., Jan, Feb)

    # Prepare input data (set all other months to 0)
    input_data = {month: 100.0}  # Example: Set rainfall for the given month to 100.0
    input_data = {m: input_data.get(m, 0.0) for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']}

    input_df = pd.DataFrame([input_data])
    predicted_rainfall = model.predict(input_df)[0]

    return predicted_rainfall

def generate_hourly_rainfall(total_rainfall):
    """Distribute total rainfall across 24 hours."""
    # Assuming rainfall is uniformly distributed across 24 hours for simplicity
    hourly_rainfall = np.random.normal(loc=total_rainfall / 24, scale=5, size=24)  # Add variability
    hourly_rainfall = np.clip(hourly_rainfall, 0, None)  # Ensure no negative rainfall
    return hourly_rainfall

def calculate_rainfall_probability(predicted_rainfall):
    """Calculate the probability of rainfall."""
    # Define realistic thresholds based on historical data
    if predicted_rainfall <= 1:  # Minimal rain
        return 10  # 10% probability
    elif predicted_rainfall <= 20:  # Low rain
        return 40  # 40% probability
    elif predicted_rainfall <= 50:  # Moderate rain
        return 70  # 70% probability
    else:  # Heavy rain
        return 90  # 90% probability

def plot_hourly_rainfall_line(date_str, hourly_rainfall):
    """Plot hourly rainfall for the given day using a line plot."""
    hours = [f"{hour}:00" for hour in range(24)]
    plt.figure(figsize=(12, 6))
    plt.plot(hours, hourly_rainfall, marker='o', linestyle='-', color='blue', label='Hourly Rainfall')
    plt.title(f"Hourly Rainfall Prediction for {date_str}")
    plt.xlabel("Hours")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rainfall_probability(probability, date_str):
    """Plot the probability of rainfall as a pie chart."""
    labels = ['Rain', 'No Rain']
    sizes = [probability, 100 - probability]
    colors = ['#1f77b4', '#ff7f0e']
    explode = (0.1, 0)  # Highlight the 'Rain' slice

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f"Rainfall Probability for {date_str}")
    plt.tight_layout()
    plt.show()

# Example prediction and graph
example_date = input("Enter the date (YYYY-MM-DD): ")
predicted_rainfall = predict_rainfall_for_date(example_date)
print(f"Predicted Total Rainfall for {example_date}: {predicted_rainfall:.2f} mm")

# Generate hourly rainfall and plot
hourly_rainfall = generate_hourly_rainfall(predicted_rainfall)
plot_hourly_rainfall_line(example_date, hourly_rainfall)

# Calculate and plot rainfall probability
rainfall_probability = calculate_rainfall_probability(predicted_rainfall)
print(f"Rainfall Probability: {rainfall_probability:.2f}%")
plot_rainfall_probability(rainfall_probability, example_date)
