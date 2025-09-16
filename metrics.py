import numpy as np

def mse(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    return np.mean((y_true-y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    numerator = np.sum((y_true-y_pred)**2)
    denominator = np.sum((y_true-np.mean(y_true))**2)
    return 1-numerator / denominator if denominator != 0 else 0

def mae(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred))

def mape(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/np.abs(y_true))

def smape(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    return np.mean((2*np.abs(y_true-y_pred))/(y_true+y_pred))

def wape(y_true, y_pred):
    y_true, y_pred=np.array(y_true), np.array(y_pred)
    numerator = np.sum(np.abs(y_true-y_pred))
    denominator = np.sum(np.abs(y_true))
    return numerator/denominator if denominator != 0 else 0

def rmsle(y_true, y_pred, c = 1.0):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((np.log(y_true+c)-np.log(y_pred+c))**2))

y_true = [100, 200, 300]
y_pred = [110, 190, 310]

print("MSE  :", mse(y_true, y_pred))
print("RMSE :", rmse(y_true, y_pred))
print("R2   :", r2(y_true, y_pred))
print("MAPE :", mape(y_true, y_pred))
print("SMAPE:", smape(y_true, y_pred))
print("WAPE :", wape(y_true, y_pred))
print("RMSLE:", rmsle(y_true, y_pred, c=1))
