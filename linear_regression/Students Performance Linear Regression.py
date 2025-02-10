import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# This section adds the necessary libraries. All the libraries we need for the operations
#we will do and the visuals to be created are added in this section

file_path = r"\datapath\StudentsPerformance.csv"

# In this field you need to write the location of your data on the computer. This line
# will hold this location with file_path. We will need it to be able to read it later.

# ADDITIONAL REMINDER: For the most efficient result, the data set can be downloaded
# and run via the link I have provided.
# kaggle link: " https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance "


StudentsPerformance = pd.read_csv(file_path)
StudentsPerformance.head()

# In this section, we read the data using the data path we just received with file_path
# and preview the first 5 lines.

sns.pairplot(StudentsPerformance)

# We draw a dot plot of each column in our data and examine the relationships between them.
# This will be very helpful for selecting the right data later on..

StudentsPerformance.drop('gender',axis=1, inplace=True)
StudentsPerformance.drop('race/ethnicity',axis=1, inplace=True)
StudentsPerformance.drop('parental level of education',axis=1, inplace=True)
StudentsPerformance.drop('lunch',axis=1, inplace=True)
StudentsPerformance.drop('test preparation course',axis=1, inplace=True)

# The Linear Regression model that we will call from Scikit-learn in the following processes
# only works with numerical data. For this reason, we delete the columns with non-numerical
# data in our data set.

StudentsPerformance.head()

# We review the final version of our data.

for column in StudentsPerformance.columns:
    print(f"{column} unique values in the column:: {StudentsPerformance[column].unique()}")

# We find unique data for all columns in the final version of our dataset. By using the
# print function in the for loop, we check for any problematic data..
# Examples of problematic data can be phrases such as '?', 'NaN' (Not a number) or 'None'.

print(StudentsPerformance.dtypes)

# It checks to see if the data type of our new data set is suitable to work. All data
# should appear as int64 at this point.

StudentsPerformance.corr()

# We apply correlation to our data set to make it easier to examine the numerical data
# we have and to clarify the relationship between them. This will also make it easier
# for us to create heatmaps later on.

plt.figure(figsize=(6, 6))
heatmap = sns.heatmap(StudentsPerformance.corr(), vmin = -1 , vmax = 1 , annot = True)
heatmap.set_title('Correlation Heatmap' , fontdict={'fontsize':12}, pad=12);

# We draw a heatmap of our data and observe the relationship between them. With this method,
# we can see the obvious relationship between them and decide our training and test data
# accordingly. Choosing the ones with high correlation between them will affect the result
# we will get in a good way. At this point, some believe that it should be greater than 0.6
# and some believe that it should be greater than 0.7. If the results are too low, the data
# set should be reviewed and actions should be taken to improve it.

x = StudentsPerformance['reading score'].values
y = StudentsPerformance['writing score'].values

# We choose the reading and writing scores with the highest correlation between them.
# Reading score will be our dependent variable and writing score will be our independent
# variable. At the end of the day, our artificial intelligence will predict the writing
# score for a student whose reading score you give.


print(x.shape)
print(type(x))
Length = len(x)
x = x.reshape(Length, 1)
print(x.shape)
print(type(x))

# Scikit-learn library can work with 2D array. Here we look at the format of our data
# and change it to a comprehensible 2D format.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4 , random_state=42)

# We call the Scikit-learn library. We distribute test and training data for our dependent
# and independent variables x and y. The test_size section decides how many percentages of
# our data will be distributed. The 0.4 we wrote here is perceived as 40% by the function and
# divides the data set accordingly. At this point, we train our artificial intelligence with
# 60% and test with 40%. This distribution may vary according to the test data and volume.
# According to some sources, 40-60 or 20-80 is the most appropriate. The random_state at the
# end ensures that the data is evenly and randomly distributed. The number 42 we use here is
# a reference to the book Hitchhiker's Guide to the Galaxy. 

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

# We call the linear regression model from Scikit-learn and train our model with the x and
#y training data we have prepared.

predictions = lm.predict(x_test)

# We then ask our model to make predictions based on this training.

sns.scatterplot(x=y_test, y=predictions)

# We compare the predictions of our model and the actual results, and to make it easier
# to do this, we plot a dot plot.

from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, lm.predict(x_train))
test_r2 = r2_score(y_test, lm.predict(x_test))

# Finally, we calculate the R2 score, which shows the success rate of this modeling,
# for both test and training data. The result with decimal fractions is our success rate
# in percentage terms. For example, if the result is 0.805, this means our model is
# approximately 80.5% accurate..

print("Training R² score:", train_r2)
print("Test R² score:", test_r2)
