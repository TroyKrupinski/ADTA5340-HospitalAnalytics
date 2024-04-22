import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats
import numpy as np
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# **Step 0: Data Loading and Initial Cleaning**
def extract_age_lower_bound(age_group_str):
    """Extracts the lower bound of an age range from a string.
    Args:
        age_group_str (str): The age group string (e.g., "20 to 29", "70 or Older").
    Returns:
        int: The lower bound of the age range, or a specified value for '70 or Older'.
    """
    if 'or Older' in age_group_str:
        return 70  # Handling "70 or Older" explicitly
    try:
        return int(age_group_str.split()[0])
    except ValueError:
        return np.nan

# Loading data with proper data types specified in dtype_dict
dtype_dict = {
    'Health Service Area': 'object',
    'Hospital County': 'object',
    'Operating Certificate Number': 'float64',
    'Facility Id': 'float64',
    'Facility Name': 'object',
    'Age Group': 'object',
    'Zip Code - 3 digits': 'object',
    'Gender': 'object',
    'Race': 'object',
    'Ethnicity': 'object',
    'Length of Stay': 'object',  # Likely the mixed-type column
    'Type of Admission': 'object',
    'Patient Disposition': 'object',
    'Discharge Year': 'int64',
    'CCS Diagnosis Code': 'int64',
    'CCS Diagnosis Description': 'object',
    'CCS Procedure Code': 'int64',
    'CCS Procedure Description': 'object',
    'APR DRG Code': 'int64',
    'APR DRG Description': 'object',
    'APR MDC Code': 'int64',
    'APR MDC Description': 'object',
    'APR Severity of Illness Code': 'int64',
    'APR Severity of Illness Description': 'object',
    'APR Risk of Mortality': 'object',
    'APR Medical Surgical Description': 'object',
    'Payment Typology 1': 'object',
    'Payment Typology 2': 'object',
    'Payment Typology 3': 'object',
    'Birth Weight': 'int64',
    'Abortion Edit Indicator': 'object',
    'Emergency Department Indicator': 'object',
    'Total Charges': 'float64',
    'Total Costs': 'float64'
}

data = pd.read_csv("HosptialDataSplit.csv", dtype=dtype_dict)

# Apply the function to the 'Age Group' column and handle non-numeric issues
data['Age'] = data['Age Group'].apply(extract_age_lower_bound)
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')  # Convert to numeric, coercing errors to NaN

# Handle missing values, if necessary
data.dropna(subset=['Age', 'Length of Stay', 'Race'], inplace=True)
majority_size = data['Race'].value_counts().max()
data_balanced = pd.concat(
    [sub_df.sample(majority_size, replace=True)
     for _, sub_df in data.groupby('Race')]
).reset_index(drop=True)

# Impute any remaining NaN values in the dataset
imputer = SimpleImputer(strategy='mean')
data_balanced[['Total Charges', 'Total Costs']] = imputer.fit_transform(data_balanced[['Total Charges', 'Total Costs']])


# Verify results
print(data_balanced['Age'].head(10))
print(data_balanced['Age'].dtype)
data_balanced['Length of Stay'] = pd.to_numeric(data_balanced['Length of Stay'], errors='coerce')
print(data_balanced['Length of Stay'].isnull().sum())

data_balanced.info()
# Handle missing data (adjust techniques as needed)
data_balanced.dropna(subset=['CCS Diagnosis Description', 'Length of Stay', 'Age'], inplace=True)  # Assuming these columns are crucial or use imputation techniques 

# **Step 1: Average Time Spent in the Hospital for Each Illness**

# Calculate average length of stay per diagnosis
# Calculate the mean length of stay, now that the column is numeric
average_los_per_diagnosis = data_balanced.groupby('CCS Diagnosis Description')['Length of Stay'].mean()

# Calculate average length of stay per diagnosis
average_los_per_diagnosis = data_balanced.groupby('CCS Diagnosis Description')['Length of Stay'].mean()

# Print the entire average_los_per_diagnosis dictionary
print(average_los_per_diagnosis)

# Display the numbers in a separate window
for diagnosis, length_of_stay in average_los_per_diagnosis.items():
    print(f"Diagnosis: {diagnosis}, Average Length of Stay: {length_of_stay}")
print('test')
# Visualize average length of stay per diagnosis
plt.figure(figsize=(10, 6))
average_los_per_diagnosis.plot(kind='bar', title='Average Length of Stay by Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Average Length of Stay')
plt.show()

top_15_avg_los = average_los_per_diagnosis.nlargest(15) 
top_15_avg_los.plot(kind='barh', title='Top 15 Average Length of Stay by Diagnosis')
plt.show()

# **Step 2: Patient Demographics and Effects on Illness Prevalence and Consequences**

# Analyze age distributions
sns.histplot(data_balanced, x='Age', bins=20, kde=True) 
plt.title('Distribution of Patient Ages')
plt.show()

# Explore illness prevalence across different demographics

diseases = data_balanced['CCS Diagnosis Description'].unique()

num_plots = math.ceil(len(diseases) / 8)
for plot_num in range(num_plots):
    start_index = plot_num * 8
    end_index = min((plot_num + 1) * 8, len(diseases))
    subset_diseases = diseases[start_index:end_index]
    
    for demographic in ['APR Risk of Mortality']:
        total_counts = data_balanced[data_balanced['CCS Diagnosis Description'].isin(subset_diseases)].groupby(demographic)['Race'].value_counts()
        plt.figure(figsize=(30, 20))
        total_counts.unstack().plot(kind='bar', stacked=True)
        plt.xticks(rotation=45, fontsize=6)  # Reduce the font size for diseases
        plt.ylabel('Count')
        plt.title(f'Count of {demographic} for Subset of Diseases: {subset_diseases}')
        plt.legend(loc='upper right')  # Add legend to the graph
        plt.show()


# Examine impact on treatment results
data_balanced.groupby(['CCS Diagnosis Description', 'APR Risk of Mortality'])['Length of Stay'].mean().unstack()

# Consider statistical tests: ANOVA to analyze differences in length of stay between demographic groups
# Perform ANOVA test

ethnic_groups = data_balanced['Ethnicity'].unique()
grouped_data = [data_balanced[data_balanced['Ethnicity'] == group]['Length of Stay'] for group in ethnic_groups]
f_statistic, p_value = stats.f_oneway(*grouped_data)

# Print the results
print("ANOVA Results:")
print("F-Statistic:", f_statistic)
print("p-value:", p_value)

# Plot the ANOVA results
plt.figure(figsize=(8, 6))
sns.boxplot(x='Ethnicity', y='Length of Stay', data=data)
plt.title('Length of Stay by Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Length of Stay')
plt.ylim(0, 100)
plt.show()

# **Step 3: Links Between Treatment Results and Diagnoses**


sns.boxplot(data=data, x='Race', y='Total Costs')
plt.title('Total Costs by Race')




# Calculate success rates by diagnosis and treatment
treatment_success_rates = data_balanced.groupby(['CCS Diagnosis Description', 'APR Medical Surgical Description'])['Length of Stay'].mean().unstack()

# Plot the treatment success rates
plt.figure(figsize=(20, 12))
num_plots = 5
subset_size = len(treatment_success_rates) // num_plots

for i in range(num_plots):
    start_index = i * subset_size
    end_index = (i + 1) * subset_size
    subset_treatment_success_rates = treatment_success_rates.iloc[start_index:end_index]
    
    plt.subplot(num_plots, 1, i+1)
    subset_treatment_success_rates.plot(kind='bar', stacked=True)
    plt.title(f'Treatment Success Rates by Diagnosis (Subset {i+1})')
    plt.xlabel('Diagnosis')
    plt.ylabel('Treatment Success Rate')
    plt.legend(title='Treatment')
    # Calculate treatment success rates by diagnosis
    treatment_success_rates = data_balanced.groupby(['CCS Diagnosis Description', 'APR Medical Surgical Description'])['Length of Stay'].mean().unstack()

    # Print the treatment success rates
    print(treatment_success_rates)
plt.tight_layout()
plt.show()
# Address missing data:
data_balanced.isnull().sum()  # Quantify missingness per column 
# **Step 4: Machine Learning Model for Length of Stay Prediction**
# Select features for the model
features = ['Age', 'Gender', 'Race', 'Total Charges', 'Total Costs', 'APR Risk of Mortality']
target = 'Length of Stay'
data_encoded = pd.get_dummies(data_balanced[features + [target]], drop_first=True)

# Split the data into training and testing sets
X = data_encoded.drop(target, axis=1)
y = data_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

# Visualize the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Length of Stay')
plt.ylabel('Predicted Length of Stay')
plt.title('Predicted vs Actual Length of Stay')
plt.show()

# Predict length of stay for specific race and total cost values
race = 'White'  # Specify the race for prediction
total_cost = data_balanced['Total Costs'].mean()  # Use the mean value of 'Total Costs' as the total cost for prediction

# Create a sample input for prediction
sample_input = pd.DataFrame({
    'Age': [50],
    'Gender_Male': [1],
    'Race_White': [1],
    'Total Charges': [total_cost],
    'Total Costs': [total_cost],
    'APR Risk of Mortality_Moderate': [1]
})

# Make the prediction
predicted_length_of_stay = model.predict(sample_input)

# Print the predicted length of stay
print(f"Predicted Length of Stay for {race} race and total cost of ${total_cost}: {predicted_length_of_stay[0]}")

print(data.shape)
