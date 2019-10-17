# Lyall
import pandas as pd

loans = pd.read_csv('loan_final313.csv')
loans = loans.drop(['id', 'issue_d', 'final_d', 'home_ownership', 'application_type', 'income_category', 'purpose',
                    'interest_payments', 'loan_condition', 'grade', 'term'], axis=1)
loans = loans.replace('munster', 0)
loans = loans.replace('leinster', 1)
loans = loans.replace('cannught', 2)
loans = loans.replace('ulster', 3)
print(loans['region'])
# Lyall
