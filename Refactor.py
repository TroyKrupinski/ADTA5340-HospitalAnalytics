import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

def extract_age_lower_bound(age_group_str):
    """Extracts the lower bound of an age range from a string."""
    if 'or Older' in age_group_str:
        return 70
    try:
        return int(age_group_str.split()[0])
    except ValueError:
        return np.nan

# Load data
dtype_dict = {
    'Health Service Area': 'object', 'Hospital County': 'object',
    'Operating Certificate Number': 'float64', 'Facility Id': 'float64',
    'Facility Name': 'object', 'Age Group': 'object', 'Zip Code - 3 digits': 'object',
    'Gender': 'object', 'Race': 'object', 'Ethnicity': 'object',
    'Length of Stay': 'object', 'Type of Admission': 'object',
    'Patient Disposition': 'object', 'Discharge Year': 'int64',
    'CCS Diagnosis Code': 'int64', 'CCS Diagnosis Description': 'object',
    'CCS Procedure Code': 'int64', 'CCS Procedure Description': 'object',
    'APR DRG Code': 'int64', 'APR DRG Description': 'object',
    'APR MDC Code': 'int64', 'APR MDC Description': 'object',
    'APR Severity of Illness Code': 'int64', 'APR Severity of Illness Description': 'object',
    'APR Risk of Mortality': 'object', 'APR Medical Surgical Description': 'object',
    'Payment Typology 1': 'object', 'Payment Typology 2': 'object', 'Payment Typology 3': 'object',
    'Birth Weight': 'int64', 'Abortion Edit Indicator': 'object',
    'Emergency Department Indicator': 'object', 'Total Charges': 'float64', 'Total Costs': 'float64'
}
data = pd.read_csv("HosptialDataSplit.csv", dtype=dtype_dict)
data['Age'] = data['Age Group'].apply(extract_age_lower_bound)
data['Length of Stay'] = pd.to_numeric(data['Length of Stay'], errors='coerce')

# Drop rows where 'Age' or 'Length of Stay' could not be converted
data.dropna(subset=['Age', 'Length of Stay', 'Race'], inplace=True)

# Balance the dataset by oversampling
majority_size = data['Race'].value_counts().max()
data_balanced = pd.concat(
    [sub_df.sample(majority_size, replace=True)
     for _, sub_df in data.groupby('Race')]
).reset_index(drop=True)

# Impute any remaining NaN values in the dataset
imputer = SimpleImputer(strategy='mean')
data_balanced[['Total Charges', 'Total Costs']] = imputer.fit_transform(data_balanced[['Total Charges', 'Total Costs']])

# Analyzing average time spent in hospital for each illness
data_balanced.groupby('CCS Diagnosis Description')['Length of Stay'].mean().nlargest(15).plot(kind='barh')
plt.title('Top 15 Average Length of Stay by Diagnosis')
plt.xlabel('Average Length of Stay')
plt.show()

# Modeling to predict Length of Stay
features = pd.get_dummies(data_balanced.drop(columns=['Length of Stay']), drop_first=True)
target = data_balanced['Length of Stay']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Linear regression model with imputation included in the pipeline
model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nMSE: {mse}, MAE: {mae}, R-squared: {r2}")

plt.scatter(X_test['Age'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Age'], y_pred, color='red', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Length of Stay')
plt.legend()
plt.show()
