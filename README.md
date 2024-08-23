# **Univariate Storage Forecasting with LSTM**
---
This repository contains code and documentation for forecasting storage data using Univariate Long Short-Term Memory (LSTM) models. The project focuses on developing a model to predict future storage values based on historical data. 

## **Objective**

The objective of this project is to accurately forecast storage levels using a univariate LSTM model, leveraging historical data for prediction. This model aims to assist in managing storage resources effectively by providing reliable forecasts.

## **Data Source**

The data used in this project is a time series dataset containing various environmental and meteorological parameters related to storage levels. The primary data file used is `Final_Lag24_storage_Data.csv`, which is a processed version of the original dataset with appropriate lag features included.

## **Stages**

1. **Data Preparation**: Involves reading the dataset, scaling the data, and splitting it into training, testing, and validation sets.
2. **Model Development**: An LSTM model is developed and trained on the prepared data.
3. **Performance Evaluation**: The model’s performance is evaluated using metrics like RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error).
4. **Result Aggregation**: The results from multiple simulations are aggregated to provide a final performance report.

## **Design**

The project is designed around a univariate time series forecasting approach using LSTM networks. The design includes data scaling, model training with early stopping, and performance evaluation on validation data.

## **Mockup**

A detailed architectural mockup of the LSTM model is provided, showing the flow of data through the network layers, including input reshaping, LSTM units, and final dense layers.

## **Tools**

- **Python Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Keras
- **Jupyter Notebook**: For interactive development and visualization
- **Google Colab**: Utilized for training models with GPU support

## **Development**

The development process includes:
1. **Data Loading and Preparation**: Loading the dataset and applying necessary transformations.
2. **Model Implementation**: Implementing the LSTM model using Keras.
3. **Training**: Training the model on the historical storage data.
4. **Evaluation**: Assessing the model’s performance on test data.

## **Pseudocode**

```python
# Load and scale data
data = load_data('Final_Lag24_storage_Data.csv')
scaled_data = scale_data(data)

# Split into train, test, and validation sets
train, test, val = split_data(scaled_data)

# Define and compile LSTM model
model = Sequential()
model.add(LSTM(20, input_shape=(train_X_reshaped.shape[1], train_X_reshaped.shape[2]), return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_X, train_y, epochs=2000, validation_data=(test_X, test_y), verbose=1, shuffle=False)

# Evaluate performance
evaluate_model(model, val_X, val_y)
```

## **Data Exploration**

Data exploration includes examining the distribution of features, checking for missing values, and analyzing the time series characteristics of the data.

## **Data Cleaning**

Data cleaning involves handling missing values, removing duplicates, and standardizing formats to ensure consistency and accuracy in the analysis.

## **Transform the Data**

Data transformation steps include scaling the features and creating lag features to capture temporal dependencies.

## **Testing**

Testing includes validating the model’s predictions against known values and conducting robustness checks across different scenarios.

## **Data Quality Tests**

- **Missing Value Check**: Ensure no missing data is present in critical features.
- **Consistency Check**: Validate that data transformations maintain consistency.

## **Visualization**

Visualization techniques used include heatmaps for correlation analysis and time series plots for trend analysis.

## **Results**

Results are saved as CSV files and include predicted values, RMSE, MAPE, and loss history. These results are then used to generate performance reports.

## **Analysis**

The analysis phase involves interpreting the model’s predictions and performance metrics to understand the reliability and accuracy of the forecasts.

## **Findings**

Key findings include the model’s ability to predict storage levels with a high degree of accuracy, as evidenced by low RMSE and MAPE values.

## **Validation**

Model validation is performed using cross-validation techniques to ensure the generalizability of the results.

## **Discovery**

The project discovered that LSTM models are particularly effective for time series forecasting, especially when handling sequential data with temporal dependencies.

## **Recommendations**

It is recommended to:
- Use the trained model for real-time storage forecasting.
- Consider further tuning of hyperparameters to improve accuracy.

## **Potential ROI**

Accurate storage forecasting can lead to significant cost savings by optimizing resource allocation and preventing shortages or overflows.

## **Potential Courses of Actions**

- **Implementation**: Deploy the model in a production environment.
- **Continuous Monitoring**: Regularly update the model with new data to maintain accuracy.

## **Conclusion**

This project demonstrates the effectiveness of LSTM networks for univariate time series forecasting. The model developed provides reliable predictions, which can be used to make informed decisions regarding storage management.

## **Google Colab Notebook Configuration**

The following configurations and code snippets are used to set up the Google Colab environment for this project:

```json
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Univariate_storage_forecasting_with_Lag24.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  }
}
```

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Load and prepare data
dataset = pd.read_csv('/content/gdrive/My Drive/PHD_THESIS/PAPER1/STORAGE_FORECAST/UVLSTM/Final_Lag24_storage_Data.csv', header=0, index_col=0)
values = dataset.values.astype('float32')
scaled, value_min, value_max = scale(values)
```

### **Model Training and Evaluation**

The model is trained with the following configuration and steps:

```python
model = KerasRegressor(build_fn=baseline_model, epochs=2000, batch_size=30, verbose=1)
history = model.fit(train_X_reshaped, train_y, validation_data=(test_X_reshaped, test_y), verbose=1, shuffle=False)

# Save the performance metrics
pd.DataFrame(trainRMSE_calc_list).to_csv('/content/gdrive/My Drive/PHD_THESIS/PAPER1/STORAGE_FORECAST/UVLSTM/Results/Performances/24Lags/rmsetrainset.csv')
```
---
