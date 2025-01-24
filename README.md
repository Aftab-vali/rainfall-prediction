# Rainfall Prediction Model

This project is a machine learning-based rainfall prediction system for Bangalore. It uses historical rainfall data to predict the total rainfall for a given date and provides detailed visualizations, including hourly rainfall distribution and rainfall probability.

## Features

- Predict total rainfall for a given date.
- Display hourly rainfall distribution for the predicted date using a line plot.
- Calculate and visualize the probability of rainfall using a pie chart.
- Intuitive and user-friendly interface for interpreting predictions.

## Dataset

The dataset used in this project includes monthly and annual rainfall data for Bangalore from 1900 to September 2024. It consists of the following columns:

- `Jan` to `Dec`: Monthly rainfall data (in mm).
- `Total`: Total annual rainfall (in mm).

## Requirements

- Python 3.8 or higher
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

To install the required libraries, run:

```bash
pip install pandas numpy scikit-learn matplotlib
