import pandas as pd
import random
import numpy as np

# Reproducibility
random.seed(42)
np.random.seed(42)

NUM_RECORDS = 100

data = []

for _ in range(NUM_RECORDS):
    age = random.randint(18, 24)
    study_hours = round(random.uniform(0.5, 8), 1)
    attendance = random.randint(55, 100)

    # Internal marks influenced by study hours and attendance + noise
    noise = random.randint(-10, 10)
    internal_marks = int((study_hours * 10) + (attendance * 0.3) + noise)

    # Clamp marks to realistic bounds
    internal_marks = max(30, min(internal_marks, 95))

    # Base rule for result
    if internal_marks >= 50 and attendance >= 65:
        result = "Pass"
    else:
        result = "Fail"

    data.append([
        age,
        study_hours,
        attendance,
        internal_marks,
        result
    ])

# Create DataFrame
columns = ["Age", "Study_Hours", "Attendance", "Internal_Marks", "Result"]
df = pd.DataFrame(data, columns=columns)

# Inject noise: flip result for ~10% of records
num_flips = int(0.1 * NUM_RECORDS)
flip_indices = np.random.choice(df.index, num_flips, replace=False)

for idx in flip_indices:
    df.at[idx, "Result"] = "Fail" if df.at[idx, "Result"] == "Pass" else "Pass"

# Inject missing values for preprocessing practice
missing_indices = np.random.choice(df.index, 8, replace=False)

for idx in missing_indices[:4]:
    df.at[idx, "Study_Hours"] = np.nan

for idx in missing_indices[4:]:
    df.at[idx, "Attendance"] = np.nan

# Save to CSV
df.to_csv("student_performance_dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
