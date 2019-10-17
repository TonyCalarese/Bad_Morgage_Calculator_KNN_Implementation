# Author: Tony Calarese, John Long, and Lyall Rogers
# Class:  DAT 330-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to another member
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).


import pandas as pd

# Lyall
loans = pd.read_csv('loan_final313.csv')
print(loans.head())

loans = loans.drop(['id', 'issue_d', 'final_d', 'home_ownership', 'application_type', 'income_category', 'purpose',
                    'interest_payments', 'loan_condition', 'grade', 'term'], axis=1)
loans = loans.replace('munster', 0)
loans = loans.replace('leinster', 1)
loans = loans.replace('cannught', 2)
loans = loans.replace('ulster', 3)
print(loans['region'])
# Lyall
