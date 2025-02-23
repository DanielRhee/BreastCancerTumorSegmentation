import pandas as pd
import os

df = pd.read_excel(os.path.join(os.path.dirname(os.getcwd()), 'Data')+"/Classification.xlsx", sheet_name="BrEaST-Lesions-USG clinical dat")
df = df.filter(['CaseID', 'Classification'])

df.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'Data')+"/Classification.csv", index=False)