import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


titanic_file_path='data/train.csv'
titanic_data=pd.read_csv(titanic_file_path)

numerical_cols=["Age", "Fare","Pclass"]
categorical_cols=["Sex","Ticket","Cabin"]

feature_columns=numerical_cols + categorical_cols
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

y=titanic_data.Survived
X=titanic_data[feature_columns]
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=21)


my_pipeline.fit(train_X, train_y)



from sklearn.metrics import accuracy_score


preds = my_pipeline.predict(val_X)

score = accuracy_score(val_y, preds)
print('Accuracy:', score)
test_data=pd.read_csv("data/test.csv")

X_test=test_data[feature_columns]
preds=my_pipeline.predict(X_test)

output=pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':preds})
output.to_csv('submission.csv',index=False)
print("it worked")
