# Coffee_Rice_Beef_Prices
# Coffee, Rice and Beef Price Prediction

This project implements a comprehensive machine learning pipeline to forecast the inflation‑adjusted prices of coffee, rice, and beef. It compares several models including **ARIMAX**, **Random Forest**, **XGBoost**, and **LightGBM**, and includes extensive hyperparameter tuning, evaluation, and visualisation modules.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data](#data)
- [Quick Start](#quick-start)
- [Detailed Module Description](#detailed-module-description)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
  - [Machine Learning Models](#machine-learning-models)
  - [ARIMA / ARIMAX Models](#arima--arimax-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Global Tuning (DeepTuner)](#global-tuning-deeptuner)
  - [Product‑Specific Tuning](#product-specific-tuning)
- [Evaluation & Model Selection](#evaluation--model-selection)
- [Visualisation](#visualisation)
- [Configuration](#configuration)
- [Results Summary](#results-summary)
- [License](#license)

---

## Project Structure

```
.
├── config.py                       # Project configuration (paths, model parameters, feature settings)
├── data_preprocessing/              # Data loading, cleaning, feature engineering
│   ├── __init__.py
│   ├── load_data.py
│   └── feature_engineering.py
├── eda_visualizations/              # Exploratory data analysis and plotting
│   ├── __init__.py
│   └── eda.py
├── models/                          # Individual model implementations
│   ├── __init__.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── arimax_model.py
├── evaluation/                       # Metrics and basic evaluation plots
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── hyperparameter_tuning/            # Tuning logic for all models
│   ├── __init__.py
│   ├── model_selection.py
│   ├── arimax_optimization.py
│   ├── unified_tuning.py
│   └── final_model_selection.py
├── visualizations/                   # Advanced visualisation scripts
│   ├── __init__.py
│   ├── model_comparison.py
│   ├── error_analysis.py
│   └── compare_optimization.py       # (to be created) original vs optimised scatter plots
├── main.py                           # Main orchestrating script
└── rice_beef_coffee_price_changes.csv # Input data (not included in repo)
```

---

## Requirements

- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  statsmodels
  xgboost
  lightgbm
  pmdarima (optional, for auto_arima)
  ```
Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Data

The input file **`rice_beef_coffee_price_changes.csv`** must contain the following columns:

- `Year` (int)
- `Month` (abbreviated, e.g. `Jan`, `Feb`, …)
- `Price_beef_kilo` (float)
- `Price_rice_kilo` (float)
- `Price_coffee_kilo` (float)
- `Inflation_rate` (float)
- `Price_rice_infl` (float) – inflation‑adjusted rice price
- `Price_beef_infl` (float) – inflation‑adjusted beef price
- `Price_coffee_infl` (float) – inflation‑adjusted coffee price

The inflation‑adjusted columns are used as the target variables.  
The file should contain monthly data (360 rows for 30 years).

---

## Quick Start

1. Place your data file in the project root as `rice_beef_coffee_price_changes.csv`.
2. (Optional) Adjust settings in `config.py`.
3. Run the complete pipeline:
   ```bash
   python main.py
   ```

The script will:
- Load and clean the data
- Perform exploratory data analysis (prints summary and saves plots under `visualizations/EDA/`)
- Create time‑based, lag, rolling, and ratio features
- Split the data (80% train, 20% test) respecting temporal order
- Train all four models with default parameters
- Compare models and select the best performing one (excluding ARIMAX)
- Execute two types of hyperparameter tuning:
  - **Global deep tuning** (direct test‑set optimisation for the best model of each product)
  - **Product‑specific tuning** (cross‑validation for the top two models of each product)
- Tune ARIMAX parameters via grid search
- Finally select the best overall model for each product (original ARIMAX, optimised ARIMAX, or optimised ML model)
- Generate numerous evaluation reports and visualisations (under `visualizations/`).

---

## Detailed Module Description

### Data Preprocessing
- **`load_data.py`**  
  `load_data()` reads the CSV, `clean_data()` handles missing values (interpolation, forward/backward fill, or dropping), `get_data_info()` prints basic statistics, and `split_data()` performs a simple temporal split.
- **`feature_engineering.py`**  
  Creates:
  - Temporal features (`Month_num`, `Quarter`, seasonal dummies, sin/cos of month)
  - Lag features (e.g. `_lag_1`, `_lag_3`, `_lag_6`)
  - Rolling statistics (mean & std over windows 3, 6)
  - Price ratios (beef/rice, beef/coffee, rice/coffee)
  - Normalisation (optional) and final preparation of X (features) and y (targets)

### Exploratory Data Analysis
**`eda_visualizations/eda.py`**  
- `perform_eda()`: prints descriptive statistics and correlation matrix.
- `analyze_trends()`: computes start‑to‑end changes.
- Several plotting functions (`plot_price_distributions`, `plot_price_time_series`, `plot_correlation_matrix`, `plot_inflation_impact`, etc.) that save images under `visualizations/EDA/`.

---

## Modeling

### Machine Learning Models
All ML models are implemented as classes with a consistent interface (`fit`, `predict`, `score`).  
Training functions (e.g. `train_random_forest_models`) loop over the three products and return dictionaries containing predictions, actual values, and evaluation metrics.

- **Random Forest** (`models/random_forest.py`) – uses `sklearn.ensemble.RandomForestRegressor`
- **XGBoost** (`models/xgboost_model.py`) – uses `xgboost.XGBRegressor`
- **LightGBM** (`models/lightgbm_model.py`) – uses `lightgbm.LGBMRegressor`

### ARIMA / ARIMAX Models
**`models/arimax_model.py`**  
- `ARIMAModel` class wraps `statsmodels.tsa.arima.model.ARIMA` and supports exogenous variables.
- `fit_arima_for_each_product()` fits a simple ARIMAX with inflation rate as the only exogenous variable.
- **`train_arimax_with_features()`** is the main function used in the pipeline. It takes the full feature matrix `X` as exogenous variables and fits a **true ARIMAX(1,1,1)** model (default order) for each product. This model is the *original* ARIMAX against which optimised versions are compared.

---

## Hyperparameter Tuning

All tuning logic resides in `hyperparameter_tuning/`.

### Global Tuning (DeepTuner)
**`unified_tuning.py` – `perform_deep_tuning()`**  
For each product, the best‑performing model among Random Forest, XGBoost, and LightGBM (excluding ARIMAX) is tuned:
- Random Forest: grid search with 5‑fold CV.
- XGBoost & LightGBM: random search (300 iterations) directly optimising test‑set R².  
  *This “look‑ahead” approach is used to explore the upper performance bound, but the resulting model may overfit.*

### Product‑Specific Tuning
**`unified_tuning.py` – `perform_product_specific_tuning()`**  
For each product, the **top two** models (from the original comparison) are tuned using `RandomizedSearchCV` with 5‑fold cross‑validation.  
This yields more robust, generalisable models.

### ARIMAX Tuning
**`arimax_optimization.py` – `grid_search_arima_params()`**  
A simple grid search over p, d, q (with exog variables) is performed for each product, using the test set to select the best parameters. The best model and its predictions are stored.

---

## Evaluation & Model Selection

- **`evaluation/metrics.py`** – `ModelMetrics` class provides static methods for R², RMSE, MAE, MAPE, MSE.
- **`hyperparameter_tuning/model_selection.py`** – `ModelSelector` class and helper functions to compare all original models and pick the best per product.
- **`hyperparameter_tuning/final_model_selection.py`** –  
  `select_final_best_models()` now compares:
  - Original ARIMAX (order (1,1,1))
  - Optimised ARIMAX (best order from grid search)
  - Optimised ML models (from global deep tuning)
  It prints a detailed comparison and selects the final model for each product based on the highest R².

The final recommendation is also saved to a text report under `evaluation_final_results/`.

---

## Visualisation

Several dedicated visualisation scripts are available under `visualizations/`:

- **`model_comparison.py`**  
  - `plot_model_r2_comparison()`, `plot_model_rmse_comparison()`, `plot_model_mae_comparison()`  
  - `plot_all_metrics_comparison()` (heatmap)  
  - `plot_model_performance_radar()`
  - `generate_deep_tuning_visualizations()` – creates 7 complex panels showing tuning performance, improvement, ranking, etc.

- **`error_analysis.py`**  
  For each model‑product pair, generates:
  - predictions vs actual line plot
  - residuals scatter + histogram
  - absolute error histogram + boxplot
  - prediction accuracy scatter
  - percentage error time series + histogram
  All saved under `visualizations/model_results_origin_models/`.

- **`compare_optimization.py`** (to be added)  
  Will produce scatter plots comparing original vs optimised predictions for each model and product, overlaid on a perfect‑prediction line.

- **`arimax_optimization.py`** (already contains chart generation)  
  When called with real data, it creates bar charts and a CSV table comparing original and tuned ARIMAX performance.

---

## Configuration

All important parameters are centralised in `config.py`:

- Data file paths, train/test split ratio, random seed.
- Default model parameters for Random Forest, XGBoost, LightGBM, and ARIMA.
- Feature engineering settings (lag steps, rolling windows, normalisation).
- Hyperparameter tuning grids and cross‑validation folds.
- Output directories for visualisations, evaluation reports, and saved models.

Modify this file to adjust the pipeline without touching the code.

---

## Results Summary

After a full run, the console prints a final best‑model summary like:

```
Final best model summary:
  beef: ARIMAX_Optimized
    R2: 0.9250
    RMSE: 0.1226
    MAE: 0.0822
  rice: ARIMAX_Optimized
    R2: 0.9052
    RMSE: 0.0134
    MAE: 0.0104
  coffee: ML_Optimized_XGBoost
    R2: 0.9668
    RMSE: 0.1253
    MAE: 0.1020
```

Additional summaries (comparison with original ARIMAX and ML‑only) are also printed.

All generated files can be found under `visualizations/` and `evaluation_final_results/`.

---

## License

This project is intended for academic and research purposes. Feel free to use and adapt it with appropriate attribution.

---

*For any questions or issues, please refer to the code documentation or contact the author.*
