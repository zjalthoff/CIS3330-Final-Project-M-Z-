import pandas as pd
import matplotlib.pyplot

df = pd.read_csv('shopping_trends.csv')

pd.set_option('display.max_columns', None)

print(df.columns)

print(df.describe())