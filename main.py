# Lyall
import pandas as pd

loans = pd.read_csv('loan_final313.csv')
loans = loans.drop(['id', 'issue_d', 'final_d', 'home_ownership', 'application_type', 'income_category', 'purpose',
                    'interest_payments', 'loan_condition', 'grade', 'term'], axis=1)
print(loans.head())
print(loans.info())
# Lyall
