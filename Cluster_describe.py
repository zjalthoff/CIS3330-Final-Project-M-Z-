import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tabulate as tb
import statsmodels.api as sm
import statsmodels.formula.api as sm_api

df = pd.read_csv('_shopping_trends_with_three_clusters.csv')
clustered_df = df[['Cluster_Number','Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location', 'Season','Subscription Status',
             'Payment Method','Previous Purchases','Frequency of Purchases']]

# print(clustered_df['Gender'].value_counts())

c1_df = clustered_df.query("Cluster_Number == 'Cluster 1'")
c2_df = clustered_df.query("Cluster_Number == 'Cluster 2'")
c3_df = clustered_df.query("Cluster_Number == 'Cluster 3'")
c4_df = clustered_df.query("Cluster_Number == 'Cluster 4'")

categorical_columns = ['Gender','Item Purchased','Category','Subscription Status','Location','Season','Payment Method','Frequency of Purchases']

print("Cluster 4 Categorical Descriptives")
for column in categorical_columns:
    count = c4_df[column].value_counts()
    # print(count)
    # print("\n")


cluster_list = [c1_df,c2_df,c3_df,c4_df]
for cluster in cluster_list:
    print(cluster.describe())
    print("\n")
    


# # # Visualizations

# Item Purchases Bar Graph
plt.xlabel('Item Purchased')
plt.xticks(rotation=90)
plt.ylabel('# of Purchases')
plt.title('Total Customer Purchases by Item')
item_counts = c1_df['Item Purchased'].value_counts()
plt.bar(x=item_counts.index,height=item_counts.values,color='red')
plt.show()

# Customer Locations Bar Graph
plt.xlabel('Customer Location')
plt.xticks(rotation=90)
plt.ylabel('# of Customers')
plt.title('State for Customer Base')
item_counts = c4_df['Location'].value_counts()
plt.bar(x=item_counts.index,height=item_counts.values,color='purple')
plt.show()


# # Age Histograms
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Customers')
plt.hist(c1_df['Age'],bins=25, edgecolor='black',color='blue') 
plt.show()

# # Previous Purchases Distribution
plt.xlabel('# of Historical Purchases')
plt.ylabel('Customer Count')
plt.title('Distribution of Historical Purchases Across Customers')
plt.hist(c1_df['Previous Purchases'],bins=25, edgecolor='black',color='blue') 
plt.show()