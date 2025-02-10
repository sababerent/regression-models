import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Importing necessary libraries. These libraries contain functions for 
# data loading, visualization, and preprocessing, which will be used 
# in later stages of the program.


file_path = r"\datapath\Hitters.csv"

# In this field you need to write the location of your data on the computer. This line
# will hold this location with file_path. We will need it to be able to read it later.

# ADDITIONAL REMINDER: For the most efficient result, the data set can be downloaded
# and run via the link I have provided.
# kaggle link: " https://www.kaggle.com/code/ahmetcankaraolan/salary-predict-with-hitters-data-set/input "

Hitters = pd.read_csv(file_path)
Hitters_orjinal = pd.read_csv(file_path)

# Here, we are reading our dataset. We are storing it twice, as 
# "Hitters_orjinal" is kept as a backup to access the original 
# data easily if needed.

Hitters.head()

# Previewing the dataset. We display the first 5 rows to understand 
# the structure of the data.

Hitters.columns

# Checking the column names of the dataset for easier processing.

Hitters.drop('League', axis=1, inplace=True)
Hitters.drop('Division', axis=1, inplace=True)
Hitters.drop('NewLeague', axis=1, inplace=True)

# Removing columns that contain categorical data since 
# Multiple Regression models work only with numerical data.

Hitters.head()

# Checking the modified dataset again to ensure that our operations 
# have been applied correctly.

Hitters.isnull().sum()

# Checking if the dataset contains missing values. This command 
# counts missing values in each column. Missing values can appear 
# as "NaN" (Not a Number), "None", or "?".

Hitters = Hitters.dropna()
Hitters.isnull().sum()

# Checking again to ensure that missing values have been successfully removed.

sns.pairplot(Hitters)

# Visualizing scatter plots to understand the relationships between 
# different features. This helps us analyze correlations before 
# applying regression.

Hitters.corr()

# Computing correlation coefficients to scale numerical relationships 
# between features within a smaller range, making patterns more interpretable.

plt.figure(figsize=(6, 6))
heatmap = sns.heatmap (Hitters.corr(), vmin =-1 , vmax =1 , annot=True)
heatmap.set_title('Correlation Heatmap' , fontdict = {'fontsize':12}, pad=12)

# Using a heatmap to visualize feature correlations. This helps us 
# identify highly correlated variables for better analysis.

for col in Hitters.columns:
    highest_highest_values = abs(Hitters.corr()[col]).nlargest( n=5) 
    print(highest_highest_values)
    for index, value in highest_highest_values.items():
      if 1 > value >=0.75:
        print(index, col, "Variables are highly correlated: ", value)

# Using a loop to find correlations greater than 0.75. Some sources 
# suggest using 0.6, 0.75, or 0.8 as a threshold for strong correlation. 
# Here, we use 0.75 as our benchmark. The results are printed.

x = Hitters[['CAtBat' , 'CHits']].values
y = Hitters['Salary'].values

# Selecting dependent (y) and independent (x) variables. In this model, 
# we aim to predict Salary based on CAtBat (Career At-Bats) and CHits (Career Hits).
#
# Explanation of selected variables:
# - CAtBat: Number of times a player has been at bat in their career.
# - CHits: Number of successful hits in the player's career.
# - Salary: The player's salary for the 1986-1987 season (in thousands).
#
# This dataset contains information about baseball players in the "Hitter" 
# position for the 1986-1987 season. Our model predicts salary based on 
# career at-bats and career hits.

print(x.shape)
print(type(x))

# Verifying the shape of X. Unlike Linear Regression, 
# Multiple Regression models can read 2D arrays, 
# allowing for multiple inputs.

Hitters['Salary'].mean()

# Calculating the average salary to compare model performance later.

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y, test_size=0.2 , random_state=42)

# Splitting the dataset into training and testing sets.
# - test_size=0.2 means 20% of the data is used for testing, while 80% is for training.
# - Different sources suggest using ratios like 70-30 or 60-40, 
#   but we chose 80-20 for this model.
# - random_state=42 ensures that the dataset is split the same way each time, 
#   allowing for reproducible results. The number 42 has no special meaning here; 
#   it's a reference to "The Hitchhiker's Guide to the Galaxy."

from sklearn.linear_model import LinearRegression
multiple_regression = LinearRegression()
multiple_regression.fit(x_train, y_train)

# Training the model using the selected features.

print(multiple_regression.coef_.round(2))
print(multiple_regression.intercept_.round(2))

# Computing regression coefficients and intercepts to assess model performance.
# This helps in understanding how changes in X values affect Y.

rent_y_predicted = multiple_regression.predict(x_test)

# Using the trained model to make salary predictions based on test data.

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, rent_y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, rent_y_predicted)
print("R2:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

# Importing necessary libraries to calculate evaluation metrics: 
# - R2 Score
# - Mean Squared Error (MSE)
# - Root Mean Squared Error (RMSE)
#
# Explanation of Metrics:
# - R2 Score (0 to 1): 1 means perfect prediction, 0 means the model explains nothing.
#   - 0.7+ → Good Model
#   - 0.3 - 0.7 → Needs Improvement
#   - Below 0.3 → Poor Model
# - MSE: Closer to 0 means better accuracy.
# - RMSE: Measures average prediction error.

# In this model, the R2 value is 0.26, meaning it needs improvement.
# The RMSE value is 365.26, which suggests a significant error in predictions.
# Since the salary values are in thousands, this error should be interpreted accordingly.


%matplotlib inline
plt.plot(y_test, label='gerçek')
plt.plot(rent_y_predicted, label='tahmin')
plt.legend()

# Visualizing the relationship between predicted and actual values 
# to analyze model accuracy.


