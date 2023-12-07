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

# # Age variable distribution
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.title('Age Distribution of Customers')
# plt.hist(df['Age'],bins=25, edgecolor='black') 
# plt.show()

# # Previous Purchases Distribution
plt.xlabel('# of Historical Purchases')
plt.ylabel('Customer Count')
plt.title('Distribution of Historical Purchases Across Customers')
plt.hist(df['Previous Purchases'],bins=25, edgecolor='black',color='purple') 
plt.show()

# # Item Category Bar Chart
# plt.xlabel('Item Purchased')
# plt.xticks(rotation=90)
# plt.ylabel('# of Purchases')
# plt.title('Total Customer Purchases by Item')
# item_counts = new_df['Item Purchased'].value_counts()
# plt.bar(x=item_counts.index,height=item_counts.values,color='green')
# plt.show()



# # # SEPARATING OBSERVATIONS BASED ON AGE

# # Proportional Age Ranges
age_bins = [18,36,54, float('inf')]
age_labels = ('18-35','36-53','54-70')
new_df['age_group'] = pd.cut(new_df['Age'], bins=age_bins, labels=age_labels, right=False)

# # Generational Age Ranges
# age_bins = [18,27,43,59,float('inf')]
# age_labels = ('Gen-Z','Millenials','Gen-X','Baby Boomers')
# new_df['age_group'] = pd.cut(new_df['Age'], bins=age_bins, labels=age_labels, right=False)

# age_bins = [18,42,58,float('inf')]
# age_labels = ('Gen-Z/Millenials','Gen-X','Baby Boomers')
# new_df['age_group'] = pd.cut(new_df['Age'], bins=age_bins, labels=age_labels, right=False)


new_df.rename(columns={'Purchase Amount (USD)':'Purchase_Amount_USD'},inplace=True)
new_df.rename(columns={'Previous Purchases':'Previous_Purchases'},inplace=True)

#print(new_df.columns)
#print(new_df['age_group'].value_counts())

# # # ANOVA Analysis - Dependent var has to be numerical/continuous, Independent var can be categorical or numerical

model1 = sm_api.ols('Purchase_Amount_USD ~ C(age_group)', data=new_df).fit() #Testing differences of purchase amounts by generation
model2 = sm_api.ols('Previous_Purchases ~ C(age_group)', data=new_df).fit() #Testing differences of amount of previous purchases by generation

anova_table1 = sm.stats.anova_lm(model1, typ=2)
anova_table2 = sm.stats.anova_lm(model2, typ=2)
# print("USD Purchase Amounts by Generation of Customer")
# print(anova_table1)
# print("\n # of Previous Purchases by Generation of Customer")
# print(anova_table2)


# # # DECISION TREE

Y = new_df['age_group']
X = new_df[[]] # what should the decision tree use to classify the age range of each customer?

