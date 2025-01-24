# rainfall-prediction
A machine learning model to predict rainfall in Bangalore using historical data (1901-2024). Includes preprocessing, training, and prediction features with month-based rainfall analysis.
# Bangalore Rainfall Prediction

This project leverages machine learning techniques to predict rainfall in Bangalore based on historical rainfall data from 1901 to 2024. The dataset includes monthly rainfall records and annual totals, allowing for comprehensive rainfall prediction using features like monthly data and climatic events.

## Features
- Predict annual rainfall using monthly rainfall data.
- Accepts specific dates as input to focus predictions on particular months.
- Implements a Random Forest Regressor for accurate predictions.
- Handles missing data and preprocesses the dataset for modeling.
- Evaluates model performance with Root Mean Squared Error (RMSE).

## Dataset
The dataset includes:
- Monthly rainfall data (January to December).
- Total annual rainfall.
- Indicators for El Niño and La Niña events.

## Requirements
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- scikit-learn
- numpy

## File Structure
```
.
├── bangalore-rainfall-data-1900-2024-sept.csv  # Dataset
├── rainfall_prediction.py                      # Main prediction script
├── README.md                                   # Project documentation
└── requirements.txt                            # Dependencies
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bangalore-rainfall-prediction.git
   cd bangalore-rainfall-prediction
   ```

2. Add the dataset file (`bangalore-rainfall-data-1900-2024-sept.csv`) to the project directory.

3. Run the prediction script:
   ```bash
   python rainfall_prediction.py
   ```

4. Input a specific date for rainfall prediction:
   ```
   Enter the date (YYYY-MM-DD): 2024-07-15
   Predicted Total Rainfall for 2024-07-15: 1050.32 mm
   ```

## Example Prediction
```python
# Example usage in code
example_date = '2024-07-15'
print(f"Predicted Total Rainfall for {example_date}: {predict_rainfall_for_date(example_date):.2f} mm")
```

## Future Enhancements
- Incorporate additional features like temperature and humidity.
- Experiment with advanced models such as Gradient Boosting or Neural Networks.
- Develop a web interface for user-friendly rainfall predictions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements.

## Author
**Mohammed Aftab Vali**

For any inquiries, contact [mohammedaftabvali@gmail.com](mail to:mohammedaftabvali@gmail.com).
