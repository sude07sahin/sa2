import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv("C:/Users/sude0/Documents/GitHub/sa2/ev/hometrain.csv")
test = pd.read_csv("C:/Users/sude0/Documents/GitHub/sa2/ev/hometest.csv")


train["SalePrice"] = np.log1p(train["SalePrice"])
y = train["SalePrice"]


train_ID = train['Id']
test_ID = test['Id']
train.drop(["Id"], axis=1, inplace=True)
test.drop(["Id"], axis=1, inplace=True)


data = pd.concat([train.drop("SalePrice", axis=1), test], axis=0, sort=False)

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())


label_enc = LabelEncoder()
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col] = label_enc.fit_transform(data[col])


scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


X_train = data_scaled.iloc[:len(train), :]
X_test = data_scaled.iloc[len(train):, :]


ridge = Ridge(alpha=10)
ridge.fit(X_train, y)
ridge_preds = ridge.predict(X_test)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y)
rf_preds = rf.predict(X_test)


xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y)
xgb_preds = xgb.predict(X_test)


final_preds = (ridge_preds + rf_preds + xgb_preds) / 3


final_preds = np.expm1(final_preds)


submission = pd.DataFrame({"Id": test_ID, "SalePrice": final_preds})
submission.to_csv("submission.csv", index=False)

print("Tahminler başarıyla oluşturuldu ve 'submission.csv' olarak kaydedildi.")
