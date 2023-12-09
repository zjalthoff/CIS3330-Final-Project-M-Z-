import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# # # Preparing the Dataframe

df = pd.read_csv('_shopping_trends_with_three_clusters.csv')
clustered_df = df[['Cluster_Number','Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location',
                    'Season','Subscription Status','Payment Method','Previous Purchases','Frequency of Purchases']]

label_encoder = LabelEncoder()
clustered_df['cluster'] = label_encoder.fit_transform(clustered_df['Cluster_Number']) #Transforms string values within cluster variable into intergers for analysis
# print(clustered_df['cluster'].value_counts())
# print(clustered_df['Cluster_Number'].value_counts())

clustered_df2 = pd.get_dummies(clustered_df,columns=['Gender','Season','Category',
                                                     'Subscription Status','Payment Method','Subscription Status']) # May have to replace Item Purchased with Category

#print(clustered_df2.columns)

Y = clustered_df2['cluster']
indep_vars = ['Gender_Male','Category_Accessories', 'Category_Clothing', 'Category_Footwear', 'Category_Outerwear']
# # # Try not to use variables that mean the same thing ,

X = clustered_df2[indep_vars]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

# # # DECISION TREE

tree_depth = 10
for i in range(1,tree_depth+1):
    cust_dt = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=i) #entropy = metric by which branches will be evaluated, randome_state = , max depth = branches depth
    cust_dt.fit(x_train,y_train)
    y_hat_DT = cust_dt.predict(x_test)
    accuracy_DT = accuracy_score(y_test,y_hat_DT)
    precision_DT = precision_score(y_test,y_hat_DT,average='weighted')
    recall_DT = recall_score(y_test,y_hat_DT,average='weighted')
    print(f"For max depth of {i}, precision is:{precision_DT}, recall is:{recall_DT}, and accuracy is:{accuracy_DT}")

# plt.figure(figsize=(15,10))
# plot_tree(cust_dt, 
#           filled=True, 
#           rounded=True, 
#           class_names=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], 
#           feature_names=X.columns)
# plt.show()

# # # RANDOM FOREST - All evaluation metrics return 1 w/ current variables (Risk of overfitting)

cust_RF = RandomForestClassifier(random_state=0)
cust_RF.fit(x_train,y_train)
y_hat_RF = cust_RF.predict(x_test)
accuracy_RF = accuracy_score(y_test,y_hat_RF)
precision_RF = precision_score(y_test,y_hat_RF,average='weighted')
recall_RF = recall_score(y_test,y_hat_RF,average='weighted')
print(f"Using the Random Forest method, precision is:{precision_RF}, recall is {recall_RF}, and accuracy is {accuracy_RF}")


# # # SVM - All evaluation metrics return 1 w/ current variables (Risk of overfitting)

cust_SVM = SVC(gamma='auto')
cust_SVM.fit(x_train,y_train)
y_hat_SVM = cust_SVM.predict(x_test)
accuracy_SVM = accuracy_score(y_test,y_hat_SVM)
precision_SVM = precision_score(y_test,y_hat_SVM,average='weighted')
recall_SVM = recall_score(y_test,y_hat_SVM,average='weighted')
print(f"Using the SVC method, precision is:{precision_SVM}, recall is {recall_SVM}, and accuracy is {accuracy_SVM}")


# # # Logistic Regression - Less than perfect, but very close to perfect w/ current variables (Risk of overfitting)

cust_LR = LogisticRegression(solver='liblinear',random_state=1)
cust_LR.fit(x_train,y_train)
y_hat_LR = cust_LR.predict(x_test)
accuracy_LR = accuracy_score(y_test,y_hat_LR)
precision_LR = precision_score(y_test,y_hat_LR,average='weighted')
recall_LR = recall_score(y_test,y_hat_LR,average='weighted')
print(f"Using the Logistic Regression method, precision is:{precision_LR}, recall is {recall_LR}, and accuracy is {accuracy_LR}")


# # # Naïve Bayes - Less than perfect, but very close to perfect w/ current variables (Risk of overfitting)

cust_NB = GaussianNB()
cust_NB.fit(x_train,y_train)
y_hat_NB = cust_NB.predict(x_test)
accuracy_NB = accuracy_score(y_test,y_hat_NB)
precision_NB = precision_score(y_test,y_hat_NB,average='weighted')
recall_NB = recall_score(y_test,y_hat_NB,average='weighted')
print(f"Using the Naïve Bayes method, precision is:{precision_NB}, recall is {recall_NB}, and accuracy is {accuracy_NB}")