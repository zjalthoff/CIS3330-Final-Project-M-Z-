import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tabulate as tb


df = pd.read_csv('shopping_trends.csv')
new_df = df[['Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location', 'Season','Subscription Status',
             'Payment Method','Previous Purchases','Frequency of Purchases']]

pd.set_option('display.max_columns', None)

#print(df.columns)


# # # DESCRIPTIVE STATISTICS


numeric_desc = new_df.describe()
numeric_desc_table = tb.tabulate(numeric_desc, headers='keys',tablefmt='pretty')
categorical_columns = ['Gender','Item Purchased','Category','Subscription Status','Location','Season','Payment Method','Frequency of Purchases']
for column in categorical_columns:
    count = new_df[column].value_counts()
    # print(count)
    # print("\n")

# print(numeric_desc_table)


# # # DESCRIPTIVE VISUALIZATIONS

#Age variable distribution
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.title('Age Distribution of Customers')
# plt.hist(df['Age'],bins=25, edgecolor='black') 
# plt.show()

# # # SEPARATING OBSERVATIONS BASED ON AGE

new_df['age_range_1'] = (new_df['Age'] >= 18) & (new_df['Age'] <= 35)
new_df['age_range_2'] = (new_df['Age'] >= 36) & (new_df['Age'] <= 53)
new_df['age_range_3'] = (new_df['Age'] >= 54) & (new_df['Age'] <= 70)

print(new_df)