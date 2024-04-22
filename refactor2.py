import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# **Step 0: Data Loading and Cleaning**

def extract_age_lower_bound(age_group_str):
    """Extracts the lower bound of an age range. Handles '70 or Older'."""
    if 'or Older' in age_group_str:
        return 70  
    try:
        return int(age_group_str.split()[0])
    except ValueError:
        return np.nan

# Load data with appropriate data types
dtype_dict = {  # Specify data types as needed
    'Health Service Area': 'object',
    'Hospital County': 'object',
    # ... other columns
}

data = pd.read_csv("HosptialDataSplit.csv", dtype=dtype_dict)

# Preprocessing
data['Age'] = data['Age Group'].apply(extract_age_lower_bound)
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Length of Stay'] = pd.to_numeric(data['Length of Stay'], errors='coerce')

# Handle missing values (choose suitable methods)
data.dropna(subset=['Age', 'CCS Diagnosis Description', 'Length of Stay'], inplace=True)

# **Step 1: Average Hospitalization Duration per Illness**

average_los_per_diagnosis = data.groupby('CCS Diagnosis Description')['Length of Stay'].mean()
top_15_avg_los = average_los_per_diagnosis.nlargest(15)

print(top_15_avg_los)  # Textual output

# Visualization
top_15_avg_los.plot(kind='barh', title='Top 15 Average Length of Stay by Diagnosis')
plt.show()

# **Step 2: Demographic Influences**

# Age distribution
sns.histplot(data, x='Age', bins=20, kde=True) 
plt.title('Distribution of Patient Ages')
plt.show()

# Disease prevalence across demographics
for demographic in ['Race', 'Zip Code - 3 digits', 'APR Risk of Mortality']: 
    diseases = data['CCS Diagnosis Description'].unique()

    for disease in diseases:  # Analyze per disease for more focused insights
        subset = data[data['CCS Diagnosis Description'] == disease]
        demo_counts = subset.groupby([demographic, 'Race']).size().unstack(fill_value=0) * 100
        demo_counts.plot(kind='bar', stacked=True, title=f'Disease: {disease}')
        plt.show()

# Impact on treatment outcomes
df_grouped = data.groupby(['CCS Diagnosis Description', 'APR Risk of Mortality'])['Length of Stay'].mean()
print(df_grouped.unstack())  # Observe differences

# Statistical Test: ANOVA
ethnic_groups = data['Ethnicity'].unique()
grouped_data = [data[data['Ethnicity'] == group]['Length of Stay'] for group in ethnic_groups]
f_statistic, p_value = stats.f_oneway(*grouped_data)
print("ANOVA Results:", f_statistic, p_value) 
 
# **Step 3: Diagnosis and Treatment Outcomes**

# Success rates (if data allows)
success_rates = data.pivot_table(
    index='Patient Disposition', 
    columns='APR Medical Surgical Description', 
    aggfunc='mean'  # Assuming a relevant metric exists
) 
success_rates.plot(kind='bar', title='Treatment Success Rates by Diagnosis')
plt.show()

# Statistical Test: Chi-Squared (if data allows)
contingency_table = pd.crosstab(data['CCS Diagnosis Description'], data['Patient Disposition'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-Squared Results:", chi2, p, dof, expected)

# **Step 4: Modeling Length of Stay (LOS)**

# Feature engineering/selection (consider medical domain knowledge)
features = ['APR MDC Description', 'Race', 'Age', 'Total Costs']  
X = pd.get_dummies(data[features], drop_first=True)
y = data['Length of Stay']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Fitting and Evaluation
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))