# loan_code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



loandata = pd.read_csv("loan1.csv")

# Look at the data summary
print(loandata.info())

print(loandata.describe())

# This data consists around 29000 observations and 8 variables
# Variables loan_status, grade, home_ownership should be a factor. Lets convert them to factor variables!

loandata['loan_status'] = loandata['loan_status'].astype('category')
loandata['grade'] = loandata['grade'].astype('category')
loandata['home_ownership'] = loandata['home_ownership'].astype('category')


# Data head
print(loandata.head())

print(loandata.info())

print(' ')
print('##################################################')
print(' ')


# lets find out number of missing values in each variable
i = list(loandata.columns)

for columns in i:
    a = loandata[columns].isnull().sum().astype('string')
    missing = columns + ': ' + a
    print(missing)

print('int_rate has 2776 missing values and emp_length has 809 missing values.')

print(' ')
print('##################################################')
print(' ')


# Lets explore the data
# loan_status

# Lets count number of people that have defaulted
print(loandata['loan_status'].value_counts())

# Lets plot number of people that have defaulted
loandata['loan_status'].value_counts().plot(kind='bar')
plt.title('Defaults vs Non-defaults')
plt.xlabel('Loan Status')
plt.show()
print(loandata.loan_status.describe())

# Out of 29000, around 3000 people have defaulted

print(' ')
print('##################################################')
print(' ')

# Grade

# Lets count how many customers belong to what grade category
print(loandata['grade'].value_counts())

# Lets plot how many customers belong to what grade category
loandata['grade'].value_counts().plot(kind='bar')
plt.title('Grade')
plt.show()
print(loandata.grade.describe())

# More than 18000 customers are from Grade A and B

print(' ')
print('##################################################')
print(' ')

# Cross tab betweeen loan_status and grade
print(pd.crosstab(loandata['loan_status'], loandata['grade']))

print(' ')
print('##################################################')
print(' ')


# Lets count how many customers own or rent a home
print(loandata['home_ownership'].value_counts())

# Lets plot how many customers own or rent a home
loandata['home_ownership'].value_counts().plot(kind='bar')
plt.title('home Ownership')
plt.show()
print(loandata.home_ownership.describe())

# Majority of the customers rent a home or are paying their mortgage

print(' ')
print('##################################################')
print(' ')

# Cross tab betweeen loan_status and home_ownership
print(pd.crosstab(loandata['loan_status'], loandata['home_ownership']))

print(' ')
print('##################################################')
print(' ')

print(' ')
print('##################################################')
print(' ')

# Cross tab betweeen loan_status and grade
print(pd.crosstab(loandata['home_ownership'], loandata['grade']))

print(' ')
print('##################################################')
print(' ')

# Now you would like to explore continuous variables to identify
# potential outliers or unexpected data structures.

# Age

loandata.age.plot(kind = 'box')
plt.show()

loandata.age.plot(kind = 'hist')
plt.show()

print(loandata.age.describe())

# there seem to be an outlier (an error) in the age column. Lets delete the outlier
# the data. A customer is 144 years old. This can't be right!
loandata = loandata[loandata.loc[:,'age'] < 100]

print(loandata.age.describe())

loandata.age.plot(kind = 'box')
plt.show()

# After deleting the outlier in the Age column. The average age of the customer is
# 28 years old. And maximum age is 94.

print(' ')
print('##################################################')
print(' ')


# Lets create a boxplot of age per grade
loandata.boxplot('age', 'grade', rot=60)

# Show the plot
plt.show()

# Lets create a boxplot of age per loan_status
loandata.boxplot('age', 'loan_status', rot=60)

# Show the plot
plt.show()



# Loan Amount

loandata.loan_amnt.plot(kind = 'box')
plt.show()

loandata.loan_amnt.plot(kind = 'hist')
plt.show()

print(loandata.loan_amnt.describe())

# Loan amount seems alright.

# Its time to deal with missing values in int_rate

# There are a few ways where we can deal with missing values:
# Delete those observations, replace with median or keep them.
# We can't delete over 2000 observations because, this could affect the model significantly.
# We can't replace missing values with median because the int_rate is assigned to
# the customer based on his/her credit history.
# Instead we keep the missing values using coarse classification

loandata['int_category'] = pd.cut(loandata['int_rate'], [0,6,12,18,loandata.int_rate.max()],
                                labels = ['0-6', '6-12', '12-18', '18+'])

loandata['int_category'] = loandata['int_category'].replace(np.nan, 'Missing')


print(loandata['int_category'].value_counts())


# Lets plot int_category
loandata['int_category'].value_counts().plot(kind='bar')
plt.title('Interest Rate Category')
plt.show()

print(loandata.int_rate.describe())

print(loandata.emp_length.describe())

# Let's do the same with emp_length

loandata['emp_cat'] = pd.cut(loandata['emp_length'], [0,15,30,45,loandata.emp_length.max()],
                                labels = ['0-15', '15-30', '30-45', '45+'])

loandata['emp_cat'] = loandata['emp_cat'].replace(np.nan, 'Missing')


print(loandata['emp_cat'].value_counts())


# Lets plot emp_cat
loandata['emp_cat'].value_counts().plot(kind='bar')
plt.title('Employment length')
plt.show()

X = loandata.drop(['loan_status', 'emp_length', 'int_rate'], axis=1)
y= loandata['loan_status']


X = pd.get_dummies(X, drop_first=True)

print(X.columns)
# Lets build the model
# 4 models will be built: Decision Tree, Random Forest, Gradient Boosting and SVM

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=21)


# Decision Tree
classifier = DecisionTreeClassifier()

#Let's train algorithm on the training data
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred1 = classifier.predict(X_test)
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

# Compute and print metrics
print("Accuracy: {}".format(classifier.score(X_test, y_test)))
print(classification_report(y_test, y_pred1))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)

#Let's train algorithm on the training data
rf.fit(X_train, y_train)

# Predict on the test set
y_pred2 = rf.predict(X_test)
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

# Compute and print metrics
print("Accuracy: {}".format(rf.score(X_test, y_test)))
print(classification_report(y_test, y_pred2))


# Random Forest
gb = GradientBoostingClassifier()

#Let's train algorithm on the training data
gb.fit(X_train, y_train)

# Predict on the test set
y_pred3 = gb.predict(X_test)
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))

# Compute and print metrics
print("Accuracy: {}".format(gb.score(X_test, y_test)))
print(classification_report(y_test, y_pred3))

# SVM
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C': [1, 10, 100],
              'SVM__gamma': [0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# Out of all 4 models, Gradient boosting is the better model to predict, with an accuracy of 88%.
