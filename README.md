# ⚽ Premier League Assist Prediction

**Course:** CP462 Data Science Project  
**Topic:** Predicting "Assists Per Game" (APG) using Machine Learning Regression Models.

## 📖 Project Overview

This project applies Data Science methodologies to analyze Premier League player statistics for the **2024/25 season**. The primary goal is to build and evaluate Machine Learning models to predict a player's **Assists Per Game (APG)** based on various offensive metrics.

The study focuses on identifying key performance indicators (KPIs) that contribute to goal-creation and handling challenges such as **Multicollinearity** and **Data Leakage**.

## 🎯 Objectives

* **Correlation Analysis:** To investigate the relationship between general player statistics (Goals, Shots, Passes) and their assist output.
* **Model Comparison:** To benchmark the performance of different regression algorithms.
* **Feature Importance:** To identify the most significant factors influencing a player's creative efficiency.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 🐍
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Statistical Analysis:** `scipy`
* **Machine Learning:** `scikit-learn` (Linear, Ridge, Lasso)

## 📊 Dataset

* **Source:** `epl_player_stats_24_25.csv`
* **Description:** Comprehensive player statistics from the current Premier League season, containing over **57 features** including Goals, Shots, Passes, Minutes Played, and Appearances.

## 🤖 Methodology & Models

We implemented and compared three regression models to handle potential overfitting and feature correlation:

1.  **Linear Regression:** Baseline model.
2.  **Ridge Regression (L2 Regularization):** To handle multicollinearity and prevent overfitting.
3.  **Lasso Regression (L1 Regularization):** To perform feature selection.

### Experimental Scenarios
To ensure the integrity of the prediction and avoid **Data Leakage**, the models were tested in two scenarios:

* **Case 1: Include Assists Features:** Uses direct assist-related stats. High accuracy but high risk of leakage (useful for descriptive analysis).
* **Case 2: Exclude Assists Features:** **(Main Focus)** Predicts APG solely based on playstyle metrics (e.g., Passing accuracy, Shot volume, Goals) to simulate real-world predictive capability.

## 💡 Key Findings

* **Top Predictors:** The features with the highest correlation to Assists are **Shots**, **Goals**, **Minutes Played**, and **Successful Passes**. This suggests that high offensive involvement translates directly to creative output.
* **Model Performance:** All three models (Linear, Ridge, Lasso) produced similar results. However, **Ridge Regression** demonstrated the best stability when dealing with highly correlated features (Multicollinearity).
* **Insight:** While offensive stats are strong predictors, caution must be exercised regarding data leakage to ensure the model allows for genuine forecasting rather than just describing past events.

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/WuttikornFunk/premier-league-assist-prediction.git
    cd premier-league-assist-prediction
    ```

2.  **Launch Jupyter Notebook**
    You can run the `.ipynb` file using **Jupyter Notebook**, **JupyterLab**, or **Google Colab**.

3.  **Load the Dataset**
    * Ensure `epl_player_stats_24_25.csv` is in the project directory.
    * If you are using Google Colab, upload the CSV file or mount your Google Drive.
    * Update the file path in the code if necessary:
        ```python
        # Example
        df = pd.read_csv('path/to/epl_player_stats_24_25.csv')
        ```

4.  **Execute**
    Run all cells to perform data cleaning, visualization, model training, and evaluation.

---

