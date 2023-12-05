import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tabulate as tb
import statsmodels.api as sm
import statsmodels.formula.api as sm_api


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

# new_df['age_range_1'] = (new_df['Age'] >= 18) & (new_df['Age'] <= 35)
# new_df['age_range_2'] = (new_df['Age'] >= 36) & (new_df['Age'] <= 53)
# new_df['age_range_3'] = (new_df['Age'] >= 54) & (new_df['Age'] <= 70)

age_bins = [18,36,54, float('inf')]
age_labels = ('18-35','36-53','54-70')
new_df['age_group'] = pd.cut(new_df['Age'], bins=age_bins, labels=age_labels, right=False)
new_df['Purchase_Amount_USD'] = new_df['Purchase Amount (USD)']

# print(new_df['age_group'].value_counts())

# # # ANOVA Analysis

model = sm_api.ols('Purchase_Amount_USD ~ C(age_group)', data=new_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# # # DECISION TREE

Y = new_df['age_group']
X = new_df[[]] # what should the decision tree use to classify the age range of each customer?

