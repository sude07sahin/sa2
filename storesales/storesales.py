# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Time series specific
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
# Load all datasets
def load_data():
    # Main datasets
    train = pd.read_csv('train.csv', parse_dates=['date'])
    test = pd.read_csv('test.csv', parse_dates=['date'])
    
    # Supplementary data
    stores = pd.read_csv('stores.csv')
    oil = pd.read_csv('oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('holidays_events.csv', parse_dates=['date'])
    transactions = pd.read_csv('transactions.csv', parse_dates=['date'])
    
    return train, test, stores, oil, holidays, transactions

train, test, stores, oil, holidays, transactions = load_data()
def explore_data(train, stores, oil, holidays, transactions):
    print("=== Train Data ===")
    print(train.info())
    print("\n=== Stores Data ===")
    print(stores.info())
    print("\n=== Oil Prices ===")
    print(oil.info())
    print("\n=== Holidays ===")
    print(holidays.info())
    print("\n=== Transactions ===")
    print(transactions.info())
    
    # Plot sales distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(train['sales'], bins=50, kde=True)
    plt.title('Sales Distribution')
    plt.show()
    
    # Plot sales over time
    daily_sales = train.groupby('date')['sales'].sum().reset_index()
    plt.figure(figsize=(15, 6))
    plt.plot(daily_sales['date'], daily_sales['sales'])
    plt.title('Total Daily Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.show()
    
    # Time series decomposition
    ts_data = train.set_index('date').groupby(pd.Grouper(freq='D'))['sales'].sum()
    decomposition = seasonal_decompose(ts_data, model='additive', period=365)
    decomposition.plot()
    plt.show()

explore_data(train, stores, oil, holidays, transactions)
def preprocess_data(train, test, stores, oil, holidays, transactions):
    # Merge store information
    train = train.merge(stores, on='store_nbr', how='left')
    test = test.merge(stores, on='store_nbr', how='left')
    
    # Add oil prices
    oil = oil.rename(columns={'dcoilwtico': 'oil_price'})
    train = train.merge(oil, on='date', how='left')
    test = test.merge(oil, on='date', how='left')
    
    # Add holiday information
    holidays = holidays[holidays['transferred'] == False]  # Remove transferred holidays
    holidays['is_holiday'] = True
    holidays = holidays[['date', 'is_holiday', 'type']].rename(columns={'type': 'holiday_type'})
    
    train = train.merge(holidays, on='date', how='left')
    test = test.merge(holidays, on='date', how='left')
    
    # Fill missing values
    train['is_holiday'] = train['is_holiday'].fillna(False)
    test['is_holiday'] = test['is_holiday'].fillna(False)
    train['holiday_type'] = train['holiday_type'].fillna('Not Holiday')
    test['holiday_type'] = test['holiday_type'].fillna('Not Holiday')
    
    # Add transactions data
    daily_transactions = transactions.groupby(['date', 'store_nbr'])['transactions'].sum().reset_index()
    train = train.merge(daily_transactions, on=['date', 'store_nbr'], how='left')
    test = test.merge(daily_transactions, on=['date', 'store_nbr'], how='left')
    
    # Feature engineering - date features
    for df in [train, test]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features (we'll need to sort first)
        df.sort_values(['store_nbr', 'family', 'date'], inplace=True)
        df['oil_price_lag_7'] = df.groupby(['store_nbr', 'family'])['oil_price'].shift(7)
        df['oil_price_lag_30'] = df.groupby(['store_nbr', 'family'])['oil_price'].shift(30)
        
        # Rolling features
        df['oil_price_rolling_7_mean'] = df.groupby(['store_nbr', 'family'])['oil_price'].transform(
            lambda x: x.rolling(7, 1).mean())
        df['oil_price_rolling_30_mean'] = df.groupby(['store_nbr', 'family'])['oil_price'].transform(
            lambda x: x.rolling(30, 1).mean())
    
    # Fill missing oil prices (simple forward fill)
    train['oil_price'] = train.groupby(['store_nbr', 'family'])['oil_price'].ffill()
    test['oil_price'] = test.groupby(['store_nbr', 'family'])['oil_price'].ffill()
    
    # Fill remaining missing values with mean
    for col in ['oil_price', 'oil_price_lag_7', 'oil_price_lag_30', 
                'oil_price_rolling_7_mean', 'oil_price_rolling_30_mean']:
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(test[col].mean())
    
    # Encode categorical variables
    categorical_cols = ['family', 'city', 'state', 'type', 'cluster', 'holiday_type']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    
    return train, test

train_processed, test_processed = preprocess_data(train, test, stores, oil, holidays, transactions)
def prepare_model_data(df):
    # Select features and target
    features = ['store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
                'oil_price', 'is_holiday', 'holiday_type', 'transactions',
                'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'is_weekend',
                'oil_price_lag_7', 'oil_price_lag_30', 'oil_price_rolling_7_mean', 'oil_price_rolling_30_mean']
    
    X = df[features]
    if 'sales' in df.columns:
        y = df['sales']
    else:
        y = None
    
    return X, y

# Prepare training data
X_train, y_train = prepare_model_data(train_processed)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['store_nbr', 'onpromotion', 'oil_price', 'transactions', 'year', 'month', 'day',
            'day_of_week', 'day_of_year', 'week_of_year', 'oil_price_lag_7', 'oil_price_lag_30',
            'oil_price_rolling_7_mean', 'oil_price_rolling_30_mean']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

# Evaluate models
results = {}
for name, model in models.items():
    rmse = evaluate_model(model, X_train, y_train, X_val, y_val)
    results[name] = rmse
    print(f"{name} RMSE: {rmse:.4f}")

# Find best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with RMSE: {results[best_model_name]:.4f}")
def time_series_model(train, test):
    # Aggregate sales by date
    ts_data = train.groupby('date')['sales'].sum().reset_index()
    ts_data = ts_data.set_index('date').asfreq('D')
    ts_data['sales'] = ts_data['sales'].fillna(0)  # Fill missing days with 0 sales
    
    # Split into train and test
    train_size = int(len(ts_data) * 0.8)
    train_ts, test_ts = ts_data[:train_size], ts_data[train_size:]
    
    # Fit SARIMA model
    try:
        sarima = SARIMAX(train_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        sarima_fit = sarima.fit(disp=False)
        
        # Forecast
        forecast = sarima_fit.get_forecast(steps=len(test_ts))
        preds = forecast.predicted_mean
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_ts, preds))
        print(f"SARIMA RMSE: {rmse:.4f}")
        
        return sarima_fit
    except Exception as e:
        print(f"Error fitting SARIMA: {e}")
        return None

sarima_model = time_series_model(train, test)
def train_final_model(X, y, best_model):
    # Train on full dataset
    best_model.fit(X, y)
    return best_model

final_model = train_final_model(X_train, y_train, best_model)

# Prepare test data
X_test, _ = prepare_model_data(test_processed)
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Make predictions
test_predictions = final_model.predict(X_test)

# Create submission file
submission = test[['id']].copy()
submission['sales'] = test_predictions
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
def plot_feature_importance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_train.columns
        importance_df = pd.DataFrame({'feature': features, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute")

plot_feature_importance(final_model, X_train)
from sklearn.model_selection import GridSearchCV

def optimize_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_

# Uncomment to run optimization (takes time)
# optimized_model = optimize_model(X_train, y_train)