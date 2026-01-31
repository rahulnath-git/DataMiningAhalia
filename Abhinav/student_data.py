import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 100 records
data = {
    'Age': np.random.randint(18, 25, 100),
    'Study_Hours': np.random.uniform(1, 10, 100).round(1),
    'Attendance': np.random.randint(60, 100, 100),
    'Internal_Marks': np.random.randint(10, 50, 100)
}

df = pd.DataFrame(data)

# Define Result logic: Pass if Study_Hours > 4 and Attendance > 75 or Internal_Marks > 30
df['Result'] = np.where((df['Study_Hours'] > 4) & (df['Attendance'] > 70) | (df['Internal_Marks'] > 35), 'Pass', 'Fail')

# Introduce a few missing values for the "Preprocessing" task
df.loc[0:4, 'Study_Hours'] = np.nan 

# Save to CSV
df.to_csv('student_data.csv', index=False)
print("Dataset 'student_data.csv' created successfully.")