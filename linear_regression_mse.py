import numpy as np
#функция которая посчитает пример из лекции (все правильно) :)
def linear_regression_mse_for_example_from_lection(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    assert X.shape == (3, 2), "Ожидаю X формы 5×5"
    assert y.shape == (3,), "Ожидаю y длины 5"

    ones = np.ones((3, 1))
    Xb = np.hstack([ones, X])

    XtX = Xb.T @ Xb
    Xty = Xb.T @ y

    beta = np.linalg.solve(XtX, Xty)
    return beta.tolist()

def linear_regression_mse(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    assert X.shape == (5, 5), "Ожидаю X формы 5×5"
    assert y.shape == (5,), "Ожидаю y длины 5"

    ones = np.ones((5, 1))
    Xb = np.hstack([ones, X])

    XtX = Xb.T @ Xb
    Xty = Xb.T @ y

    beta = np.linalg.solve(XtX, Xty)
    return beta.tolist()

X1=[[23, 0.5, 1, 2, 3],[35, 1, 4, 5, 6], [18, 0, 13, 45, 5], [3, 13, 41, 5, 6], [1, 0, 1, 45, 5]]
y1=[55,100,45, 1, 51]
print(linear_regression_mse(X1, y1))

X2=[[23, 0.5],[35, 1], [18, 0]]
y2=[55,100,45]
print(linear_regression_mse_for_example_from_lection(X2, y2))