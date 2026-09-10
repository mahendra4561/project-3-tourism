import pandas as pd

transaction = pd.read_excel("Transaction.xlsx")
print(transaction.columns)
print(transaction.head())
