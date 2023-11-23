# battery_model

## 1. Introduction
This is an overview of my Python script that reads battery data from a CSV file, processes it using the Pandas library, performs data analysis, and builds a machine learning model to predict the State of Charge (SoC) of a battery. The script uses Lazy Predict to quickly evaluate various regression models and employs Extra Trees Regressor for the final prediction.

## 2. Data Preprocessing
The script starts by importing the required libraries, including Pandas and CSV. It then reads a CSV file named "TripA01.csv" and extracts the header row. The header is converted to a list, and the data is processed to replace semicolons and newline characters. The data is then loaded into a Pandas DataFrame, and the resulting DataFrame is saved to a new CSV file named "battery_heating_data.csv."

```python
import pandas as pd
import csv

# Open the CSV file
with open("battery_dataset/battery_heating_data/TripA01.csv", "r") as dataset:
    csv_reader = csv.reader(dataset)
    
    # Get the header row
    header = next(csv_reader)
    
    # Convert the header to a list and remove semicolons and new line character
    header_list = header.replace(';', ',').replace('\n', ',').split(',')
    
    # Print the header row
    print("Header:", header_list)
    
    rows = []
    
    # Process each row and append to the rows list
    for row in csv_reader:
        rows.append(row.replace(';', ',').replace('\n', ',').split(','))

# Convert rows and header to a Pandas DataFrame
data_frame = pd.DataFrame(rows, columns=header_list)

# Save DataFrame to CSV
data_frame.to_csv("battery_heating_data.csv", index=False)
print(data_frame.shape)
```

## 3. Data Analysis
The script proceeds to analyze the "SoC [%]" values in the dataset. It reads the CSV file created earlier and prints the shape of the DataFrame and descriptive statistics for the "SoC [%]" column.

```python
import pandas as pd

# Read the CSV file into a Pandas DataFrame
battery_heating_data = pd.read_csv("battery_heating_data.csv")
print(battery_heating_data.shape)

# Describe 'SoC [%]' values
soc_description = battery_heating_data['SoC [%]'].describe()
print(soc_description)
```

## 4. Data Preparation
The script drops the last column from the DataFrame as it contains NaN values. It then separates the features (X) from the target variable (y) and splits the dataset into training and testing sets using `train_test_split` from scikit-learn.

```python
# Drop the last column containing NaN values
battery_heating_data = battery_heating_data.drop(battery_heating_data.columns[-1], axis=1)
print(battery_heating_data.shape)

# Generate X dataset
X = battery_heating_data.drop('SoC [%]', axis=1)
print(X.shape)

# Generate y dataset
y = battery_heating_data[['SoC [%]']]
print(y.shape)

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=117)
```

## 5. Model Evaluation Using Lazy Predict
The script utilizes Lazy Predict to quickly evaluate various regression models and print their performance metrics.

```python
# Build model using Lazy Predict
from lazypredict.Supervised import LazyRegressor

# Initialize and run LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Print the list of models and their performance metrics
print(models)
```

## 6. Building and Evaluating the Final Model
The script builds a machine learning model using the Extra Trees Regressor, makes predictions on the test set, and calculates performance metrics such as Mean Squared Error (MSE), R-Squared, and Root Mean Squared Error (RMSE). It then visualizes the results by creating a scatter plot.

```python
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

# Create and train the ExtraTreesRegressor model
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the performance metrics
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
rmse = math.sqrt(mse)
print(f"Mean Squared Error: {mse}, R-Squared: {r_squared}, RMSE: {rmse}")

# Visualize results by creating a scatter plot
import numpy as np
coeffs = np.polyfit(y_test.values.flatten(), y_pred, 1)
fit_func = np.poly1d(coeffs)

plt.scatter(y_test.values, y_pred)
plt.plot(y_test, fit_func(y_test), 'g')
plt.savefig('soh_model_scatter_plot.png')
```
## 7. Result
After building and evaluating the Extra Trees Regressor model, the following performance metrics were obtained:

Mean Squared Error: 2.032596134790866e-05 ≈ 0
R-Squared: 0.999991357213117 ≈ 1
Root Mean Squared Error: 0.004508432249453092 ≈ 0
These results suggest an excellent fit of the model to the data, with very low mean squared error and high R-squared, indicating a strong correlation between the predicted and actual values of the State of Charge. The root mean squared error provides a small measure of the average prediction error, emphasizing the accuracy of the model's predictions. The scatter plot visualizes the relationship between the predicted and actual values, further confirming the model's effectiveness.