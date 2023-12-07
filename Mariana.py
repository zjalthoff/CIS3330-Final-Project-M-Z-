import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


train_file = '_shopping_trends_with_three_clusters.csv'

df_train = pd.read_csv(train_file)
df_train = pd.get_dummies(df_train,columns=['Gender', 'Item Purchased', 'Category', 'Subscription Status','Season', 'Payment Method',
                       'Frequency of Purchases'])


# clustered_df = df_train[['Cluster_Number','Age','Gender','Item Purchased','Category','Purchase Amount (USD)', 'Season',
#                   'Subscription Status', 'Payment Method', 'Previous Purchases', 'Frequency of Purchases']]

# cluster_1 = clustered_df.query("Cluster_Number =='Cluster 1'")
# cluster_2 = clustered_df.query("Cluster_Number =='Cluster 2'")
# cluster_3 = clustered_df.query("Cluster_Number =='Cluster 3'")
# cluster_4 = clustered_df.query("Cluster_Number =='Cluster 4'")

# categorical_columns = ['Gender', 'Item Purchased', 'Category', 'Subscription Status', 'Location', 'Season', 'Payment Method',
#                        'Frequency of Purchases']
# for column in categorical_columns:
#     count = cluster_2[column].value_counts()
#     print(count)
#     print("\n")

label_encoder = LabelEncoder()
df_train['cluster'] = label_encoder.fit_transform(df_train['Cluster_Number'])
print(df_train['cluster'].value_counts())
print(df_train['Cluster_Number'].value_counts())

# print(df_train.columns)
independent_variables = ['Gender_Female',
       'Gender_Male', 'Item Purchased_Backpack', 'Item Purchased_Belt',
       'Item Purchased_Blouse', 'Item Purchased_Boots', 'Item Purchased_Coat',
       'Item Purchased_Dress', 'Item Purchased_Gloves',
       'Item Purchased_Handbag', 'Item Purchased_Hat', 'Item Purchased_Hoodie',
       'Item Purchased_Jacket', 'Item Purchased_Jeans',
       'Item Purchased_Jewelry', 'Item Purchased_Pants',
       'Item Purchased_Sandals', 'Item Purchased_Scarf',
       'Item Purchased_Shirt', 'Item Purchased_Shoes', 'Item Purchased_Shorts',
       'Item Purchased_Skirt', 'Item Purchased_Sneakers',
       'Item Purchased_Socks', 'Item Purchased_Sunglasses',
       'Item Purchased_Sweater', 'Item Purchased_T-shirt',
       'Category_Accessories', 'Category_Clothing', 'Category_Footwear',
       'Category_Outerwear', 'Subscription Status_No',
       'Subscription Status_Yes', 'Season_Fall', 'Season_Spring',
       'Season_Summer', 'Season_Winter', 'Payment Method_Bank Transfer',
       'Payment Method_Cash', 'Payment Method_Credit Card',
       'Payment Method_Debit Card', 'Payment Method_PayPal',
       'Payment Method_Venmo', 'Frequency of Purchases_Annually',
       'Frequency of Purchases_Bi-Weekly',
       'Frequency of Purchases_Every 3 Months',
       'Frequency of Purchases_Fortnightly', 'Frequency of Purchases_Monthly',
       'Frequency of Purchases_Quarterly', 'Frequency of Purchases_Weekly']

X = df_train[independent_variables]
y = df_train['Cluster_Number']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)

depth = []
tree_depth = 8
for i in range(1, tree_depth):
    clf_tree = DecisionTreeClassifier(criterion="entropy", random_state= 100, max_depth= i)
    clf_tree.fit(X_train, y_train)
    yhat = clf_tree.predict(X_test)
    depth.append(accuracy_score(y_test,yhat))

    print(f"For max depth {i}: {accuracy_score(y_test,yhat)}")

plt.figure(figsize=(15,10))
plot_tree(clf_tree, 
          filled=True, 
          rounded=True, 
          class_names=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], 
          feature_names=X.columns)
plt.show()