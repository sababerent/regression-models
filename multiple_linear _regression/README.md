# Multiple Linear Regression with Hitters Dataset

## 📌 Project Overview
This project applies **Multiple Linear Regression** to the **Hitters dataset**, which contains information about baseball players from the 1986-1987 season. The goal is to predict a player's salary based on career statistics.

## 📂 Dataset Information
- **Source:** The dataset was obtained from **Kaggle**.
- **Kaggle Link:** [Hitters Dataset](https://www.kaggle.com/code/ahmetcankaraolan/salary-predict-with-hitters-data-set/input)
- **Key Features Used in This Model:**
  - `CAtBat`: Number of times a player has been at bat in their career.
  - `CHits`: Number of successful hits in the player's career.
  - `Salary`: The player's salary for the 1986-1987 season (in thousands).

## 🛠️ Technologies Used
- **Python 3.x**
- **pandas** (for data manipulation)
- **numpy** (for numerical operations)
- **matplotlib & seaborn** (for data visualization)
- **scikit-learn** (for regression modeling and evaluation metrics)

## 🚀 How to Run the Project
### 1️⃣ Install Dependencies
Before running the project, install the required libraries using:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2️⃣ Set Up the Dataset
Download the dataset from Kaggle and place it in the appropriate folder. Update the **file path** in the script:
```python
file_path = "C:/path_to_your_data/Hitters.csv"
```

### 3️⃣ Run the Python Script
Execute the script to train the model and evaluate its performance:
```sh
python Hitters_Multiple_Regression.py
```

## 📊 Model Evaluation
The model's accuracy is assessed using:
- **R2 Score:** Indicates how well the model explains the variance in salary.
- **MSE (Mean Squared Error):** Measures the average squared error.
- **RMSE (Root Mean Squared Error):** Helps interpret error in salary prediction.

### 🔎 Model Results
- **R2 Score:** 0.26 (indicating room for improvement)
- **RMSE:** 365.26 (significant prediction error, as salary is measured in thousands)

## 🏆 Key Insights
- The model's accuracy can be improved by adding more features or using different regression techniques.
- Visualizations like **heatmaps and scatter plots** help understand feature relationships.
- The model currently uses only two predictors (`CAtBat` and `CHits`), but adding more features could enhance predictions.

## 📝 Notes
- This model is built for educational purposes and can be further optimized.
- Feel free to experiment with different features and regression techniques!

## 🔗 References
- Kaggle dataset: [Hitters Dataset](https://www.kaggle.com/code/ahmetcankaraolan/salary-predict-with-hitters-data-set/input)
- Scikit-learn documentation: [scikit-learn.org](https://scikit-learn.org/)

---
🚀 **Happy Coding!**

