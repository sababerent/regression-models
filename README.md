# Regression Models

## Overview
This repository contains implementations of various regression models, starting with a simple **Linear Regression Model**. The models are implemented in **Python** using libraries such as **pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn**.

## Dataset
- The dataset used in this project is sourced from **Kaggle**.
- **Dataset Name:** [Student Study Performance](https://www.kaggle.com/datasets/bhavikjikadara/student-study-performance)
- **Download Instructions:**
  - Download the dataset from the Kaggle link above.
  - Place the dataset in the project directory.
  - Update the `file_path` variable in the script with the correct file location.

## Project Structure
```
regression-models/
│── LinearStudents_Performance_Linear_Regression.py  # Main script for Linear Regression
│── data/                                            # Directory for dataset (not included in repo)
│── README.md                                        # Project documentation
```

## Requirements
Ensure you have the following dependencies installed before running the script:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Script
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/regression-models.git
   ```
2. Navigate to the project directory:
   ```bash
   cd regression-models
   ```
3. Run the Python script:
   ```bash
   python LinearStudents_Performance_Linear_Regression.py
   ```

## Explanation of the Model
1. **Data Preprocessing:**
   - Load the dataset.
   - Remove non-numerical columns (e.g., gender, parental education level) since the linear regression model requires numerical data.
   - Check for missing or inconsistent values.
2. **Exploratory Data Analysis (EDA):**
   - Plot scatter plots of different variables to analyze relationships.
   - Generate a heatmap to visualize correlations between numerical features.
3. **Feature Selection:**
   - The model uses **Reading Score** as the independent variable (`x`) and **Writing Score** as the dependent variable (`y`).
   - These features were chosen based on their strong correlation.
4. **Splitting Data:**
   - The dataset is split into **training (60%)** and **testing (40%)** subsets.
5. **Model Training:**
   - A linear regression model is trained on the training data.
6. **Model Evaluation:**
   - Predictions are made on the test set.
   - The **R² score** is calculated to measure model accuracy.

## Results
- The **R² score** shows how well the model fits the data. A higher R² score indicates better performance.
- The scatter plot visualizes the actual vs. predicted values.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## Future Improvements
- Implement additional regression models (e.g., Multiple Regression, Polynomial Regression).
- Enhance feature selection and data preprocessing.
- Experiment with different datasets.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Contact
For any questions or suggestions, feel free to reach out:
- **GitHub:** sababerent (https://github.com/sababerent)
- **Email:** berent.93@gmail.com

