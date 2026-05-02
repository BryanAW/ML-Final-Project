# Predicting Used Car Market Values Using Machine Learning Regression Models

**Author:** Bryan Witherspoon  
**Course:** ITCS 3156 - Introduction to Machine Learning, Spring 2026  
**University of North Carolina at Charlotte**

## About

This is my final project for ITCS 3156, in which I built two regression models from scratch to predict used-car prices based on vehicle attributes such as year, manufacturer, mileage, condition, and more. The goal was to see which factors matter most when pricing a used car and to compare how well different ML approaches handle this kind of real-world data.

## Models Used

- **Ordinary Least Squares (OLS)** — A linear regression model that computes optimal weights using the closed-form solution `w = (X^T X)^{-1} X^T y`. Covered in Module 6 (Curve Fitting and Regularization).
- **k-Nearest Neighbors (KNN) Regression** — A non-parametric model adapted from the classification version we learned in Module 3. Instead of majority voting, it averages the target values of the k nearest neighbors to make a prediction.

Both models were implemented from scratch using NumPy (no sklearn model classes) and followed the structure of our homework assignments.

## Results

| Model | Test RMSE | Test R² |
|-------|-----------|---------|
| OLS | $6,638.64 | 0.7283 |
| KNN (k=10) | $5,848.07 | 0.7681 |

KNN actually outperformed OLS on the test set, which was surprising. It suggests that used car pricing exhibits non-linear patterns that a distance-based approach can capture better than a straight linear model.

## Dataset

The dataset is the **Craigslist Cars/Trucks** dataset from Kaggle. It's too large (~1.4 GB) to include in this repo, so you'll need to download it yourself:

1. Go to [https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. Click **Download** (you'll need a free Kaggle account)
3. Place the `vehicles.csv` file in the same directory as the notebook

## How to Run

1. Clone this repo or download the notebook
2. Download `vehicles.csv` from the Kaggle link above and put it in the same folder
3. Open `UsedCarPricePrediction.ipynb` in JupyterLab or Jupyter Notebook
4. Run all cells (`Kernel → Restart Kernel and Run All Cells`)

**Note:** The KNN section takes a few minutes to run since it computes distances across thousands of samples for each of the 5 k-values tested.

## Dependencies

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (used for `LabelEncoder` and `StandardScaler` only — models are coded from scratch)
