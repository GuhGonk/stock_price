split_date = X1.index[round(len(X)*0.8)]  # 80/20 
X_train, X_test = X1.loc[X1.index <= split_date], X1.loc[X1.index > split_date]
Y_train, Y_test = Y1.loc[Y1.index <= split_date], Y1.loc[Y1.index > split_date]

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

xgb_model = xgb.XGBRegressor(n_estimators = 1000, learning_rate = 0.01, random_state = 42)
xgb_model.fit(X_train, Y_train, 
          eval_set = [(X_train, Y_train), (X_test, Y_test)],
          early_stopping_rounds = 50,
          verbose = 100)

Y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(Y_test, Y_pred)
xgb_rmse = np.sqrt(xgb_mse)
print(f"Root Mean Squared Error: {xgb_rmse}")