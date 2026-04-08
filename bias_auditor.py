import pandas as pd

# 1. Load the data
df = pd.read_csv('loan_data.csv')

# 2. Calculate Approval Rates by Age Group
# This is a 'Disparate Impact' analysis
summary = df.groupby('age_group')['loan_status'].value_counts(normalize=True).unstack()
summary['Approval_Rate'] = summary['Approved'] * 100

print("--- Approval Rates by Demographic ---")
print(summary[['Approval_Rate']])

# 3. Calculate the Adverse Impact Ratio (AIR)
# AIR = (Protected Group Approval Rate) / (Control Group Approval Rate)
over_40_rate = summary.loc['over_40', 'Approval_Rate']
under_40_rate = summary.loc['under_40', 'Approval_Rate']

air = over_40_rate / under_40_rate

print(f"\nAdverse Impact Ratio (AIR): {air:.2f}")

# 4. Compliance Flag (The '80% Rule')
# Federal oversight often uses 0.80 as a threshold for potential bias
if air < 0.80:
    print("RESULT: Potential Bias Detected. AIR is below 0.80 threshold.")
    print("ACTION: Move to matched-pair file review for UDAAP compliance.")
else:
    print("RESULT: No immediate evidence of disparate impact based on age.")

import matplotlib.pyplot as plt

summary['Approval_Rate'].plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Loan Approval Rates: Under 40 vs Over 40')
plt.ylabel('Approval %')
plt.axhline(y=80, color='r', linestyle='--', label='80% Threshold')
plt.show()