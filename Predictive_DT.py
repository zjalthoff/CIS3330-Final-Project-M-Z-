import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('_shopping_trends_with_three_clusters.csv')
clustered_df = df[['Cluster_Number','Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location',
                    'Season','Subscription Status','Payment Method','Previous Purchases','Frequency of Purchases']]

label_encoder = LabelEncoder()
clustered_df['cluster'] = label_encoder.fit_transform(clustered_df['Cluster_Number'])
# print(clustered_df['cluster'].value_counts())
# print(clustered_df['Cluster_Number'].value_counts())

clustered_df2 = pd.get_dummies(clustered_df,columns=['Season','Category',
                                                     'Subscription Status','Payment Method','Subscription Status']) # May have to replace Item Purchased with Category

print(clustered_df2.columns)

Y = clustered_df2['cluster']
indep_vars = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter','Category_Accessories',
       'Category_Clothing', 'Category_Footwear', 'Category_Outerwear',
        'Subscription Status_No', 'Subscription Status_Yes','Payment Method_Cash']
X = clustered_df2[indep_vars] # what should the decision tree use to classify the age range of each customer?

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

# # # DECISION TREE ATTEMPT

tree_depth = 25
for i in range(1,tree_depth+1):
    cust_dt = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=i) #entropy = metric by which branches will be evaluated, randome_state = , max depth = branches depth
    cust_dt.fit(x_train,y_train)
    y_hat_DT = cust_dt.predict(x_test)
    accuracy_DT = accuracy_score(y_test,y_hat_DT)
    precision_DT = precision_score(y_test,y_hat_DT,average='weighted')
    f_1_DT = f1_score(y_test,y_hat_DT,average='weighted')
    print(f"For max depth of {i}, precision is:{precision_DT}, F1 is:{f_1_DT}, and accuracy is:{accuracy_DT}")