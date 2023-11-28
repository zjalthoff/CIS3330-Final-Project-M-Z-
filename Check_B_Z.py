import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('shopping_trends.csv')
new_df = df[['Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location', 'Season','Subscription Status','Payment Method','Previous Purchases','Frequency of Purchases']]

pd.set_option('display.max_columns', None)

#print(df.columns)

numeric_desc = new_df.describe()
categorical_desc = new_df['Gender'].value_counts(),df['Item Purchased'].value_counts(),df['Category'].value_counts(),df['Subscription Status'].value_counts(),df['Location'].value_counts(),df['Season'].value_counts(),df['Payment Method'].value_counts(),df['Frequency of Purchases'].value_counts()

print(numeric_desc)
print(categorical_desc)

# # VISUALIZATIONS

