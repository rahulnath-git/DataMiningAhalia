import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "Age": np.random.randint(17, 26, 100),
    "Study_Hours": np.round(np.random.uniform(1, 8, 100), 1),
    "Attendance": np.random.randint(60, 100, 100),
    "Internal_Marks": np.random.randint(30, 100, 100)
}

df = pd.DataFrame(data)

df["Result"] = np.where(
    (df["Study_Hours"] >= 4) &
    (df["Attendance"] >= 75) &
    (df["Internal_Marks"] >= 50),
    "Pass", "Fail"
)

df.to_csv("student_data.csv", index=False)

print("Dataset created successfully")
