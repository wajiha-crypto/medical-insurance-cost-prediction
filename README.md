# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges based on a person's age, BMI, smoking status, number of children, sex, and region.

This project was completed as part of a technical content writing / data science internship at [Big Brains](https://bigbrainss.com/).

## Problem Statement

Insurance charges vary significantly between individuals based on personal and lifestyle factors. This project builds a regression model to estimate insurance costs from these factors, helping understand which characteristics drive cost the most.

## Dataset

- **Source:** [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Format:** CSV
- **Records:** 1,338 rows
- **Features:** age, sex, bmi, children, smoker, region
- **Target variable:** charges

## Workflow

### 1. Data Collection
Loaded the dataset using Pandas and explored its structure, data types, and basic statistics.

### 2. Data Cleaning & Preprocessing
- Checked for missing values (none found)
- Identified and removed 1 duplicate row
- Encoded categorical variables (`sex`, `smoker` via label mapping, `region` via one-hot encoding)

### 3. Exploratory Data Analysis
- Correlation analysis showed `smoker` (0.79), `age` (0.30), and `bmi` (0.20) as the strongest predictors of charges
- Visualizations: correlation heatmap, boxplot (charges by smoking status), scatter plots (age vs charges, bmi vs charges), and a distribution histogram of charges
- Key finding: smokers pay roughly 4-5x more than non-smokers on average, and the effect of BMI on cost is much stronger for smokers

### 4. Model Selection & Training
- **Model used:** Multiple Linear Regression
- Split data into 80% training / 20% testing (`random_state=42`)
- Trained using scikit-learn's `LinearRegression`

### 5. Model Evaluation

| Metric | Value |
|---|---|
| MAE | *(insert your value)* |
| MSE | *(insert your value)* |
| RMSE | *(insert your value)* |
| R² Score | *(insert your value)* |

### 6. Prediction Application
Built an interactive prediction tool (using ipywidgets) where a user can input age, sex, BMI, children, smoker status, and region to get an estimated insurance cost.

**Example predictions:**
- 25-year-old, non-smoker, BMI 22.8 → ~$1,540
- 45-year-old, smoker, BMI 35.7 → ~$34,759

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- ipywidgets

## Limitations
- Linear Regression assumes linear relationships; it may not fully capture more complex interactions between features
- Dataset size (1,338 rows) is relatively small
- Model does not account for pre-existing medical conditions or other health data not present in the dataset

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets`
3. Open `insurance_prediction.ipynb` in Jupyter or Google Colab
4. Run all cells in order

## Project Structure
```
medical-insurance-cost-prediction/
├── insurance.csv
├── insurance_prediction.ipynb
├── README.md
```
