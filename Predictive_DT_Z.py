import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
indep_vars = ['Gender_Male','Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter','Category_Accessories', # Likelihood of overfitting greatly increased with the addition of gender variables
       'Category_Clothing', 'Category_Footwear', 'Category_Outerwear', 'Subscription Status_Yes', 'Payment Method_Cash']
# # # Try not to use variables that mean the same thing

X = clustered_df2[indep_vars]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

# # # DECISION TREE

tree_depth = 25
for i in range(1,tree_depth+1):
    cust_dt = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=i) #entropy = metric by which branches will be evaluated, randome_state = , max depth = branches depth
    cust_dt.fit(x_train,y_train)
    y_hat_DT = cust_dt.predict(x_test)
    accuracy_DT = accuracy_score(y_test,y_hat_DT)
    precision_DT = precision_score(y_test,y_hat_DT,average='weighted')
    f_1_DT = f1_score(y_test,y_hat_DT,average='weighted')
    print(f"For max depth of {i}, precision is:{precision_DT}, F1 is:{f_1_DT}, and accuracy is:{accuracy_DT}")


# # # RANDOM FOREST - All evaluation metrics return 1 w/ current variables (Risk of overfitting)

# cust_RF = RandomForestClassifier(random_state=0)
# cust_RF.fit(x_train,y_train)
# y_hat_RF = cust_RF.predict(x_test)
# accuracy_RF = accuracy_score(y_test,y_hat_RF)
# precision_RF = precision_score(y_test,y_hat_RF,average='weighted')
# f_1_RF = f1_score(y_test,y_hat_RF,average='weighted')
# print(f"Using the Random Forest method, precision is:{precision_RF}, F1 is {f_1_RF}, and accuracy is {accuracy_RF}")


# # # SVM - All evaluation metrics return 1 w/ current variables (Risk of overfitting)

# cust_SVM = SVC(gamma='auto')
# cust_SVM.fit(x_train,y_train)
# y_hat_SVM = cust_SVM.predict(x_test)
# accuracy_SVM = accuracy_score(y_test,y_hat_SVM)
# precision_SVM = precision_score(y_test,y_hat_SVM,average='weighted')
# f_1_SVM = f1_score(y_test,y_hat_SVM,average='weighted')
# print(f"Using the Random Forest method, precision is:{precision_SVM}, F1 is {f_1_SVM}, and accuracy is {accuracy_SVM}")


# # # Logistic Regression - Less than perfect, but very close to perfect w/ current variables (Risk of overfitting)

# cust_LR = LogisticRegression(solver='liblinear',random_state=1)
# cust_LR.fit(x_train,y_train)
# y_hat_LR = cust_LR.predict(x_test)
# accuracy_LR = accuracy_score(y_test,y_hat_LR)
# precision_LR = precision_score(y_test,y_hat_LR,average='weighted')
# f_1_LR = f1_score(y_test,y_hat_LR,average='weighted')
# print(f"Using the Random Forest method, precision is:{precision_LR}, F1 is {f_1_LR}, and accuracy is {accuracy_LR}")


# # # Naïve Bayes - Less than perfect, but very close to perfect w/ current variables (Risk of overfitting)

# cust_NB = GaussianNB()
# cust_NB.fit(x_train,y_train)
# y_hat_NB = cust_NB.predict(x_test)
# accuracy_NB = accuracy_score(y_test,y_hat_NB)
# precision_NB = precision_score(y_test,y_hat_NB,average='weighted')
# f_1_NB = f1_score(y_test,y_hat_NB,average='weighted')
# print(f"Using the Random Forest method, precision is:{precision_NB}, F1 is {f_1_NB}, and accuracy is {accuracy_NB}")