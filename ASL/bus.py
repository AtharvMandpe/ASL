import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('bus.csv')


# Split the dataset into train and test sets
data_train = data.sample(frac=0.8, random_state=1)
data_test = data.drop(data_train.index)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(data_train.drop(['Start_time', 'End_time', 'Location', 'Delay', 'Day'], axis=1), data_train['End_time'])
lr_prediction = lr_model.predict(data_test.drop(['Start_time', 'End_time', 'Location', 'Delay', 'Day'], axis=1))
lr_rmse = mean_squared_error(data_test['End_time'], lr_prediction, squared=False)

# Multiple Linear Regression
mr_model = LinearRegression()
mr_model.fit(data_train[['Existing_Passengers', 'Age']], data_train['End_time'])
mr_prediction = mr_model.predict(data_test[['Existing_Passengers', 'Age']])
mr_rmse = mean_squared_error(data_test['End_time'], mr_prediction, squared=False)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data_train[['Existing_Passengers', 'Age']])
model2 = LinearRegression()
model2.fit(X_poly, data_train['End_time'])

X_poly_test = poly.transform(data_test[['Existing_Passengers', 'Age']])
prediction_pr = model2.predict(X_poly_test)
rmse2 = mean_squared_error(data_test['End_time'], prediction_pr, squared=False)

# Compare RMSE values
rmse_df = pd.DataFrame({
    'RMSE': [lr_rmse, mr_rmse, rmse2],
    'Model': ['Linear Regression', 'Multiple Regression', 'Polynomial Regression']
})
print(rmse_df)
