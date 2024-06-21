from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

def calc_r2mse(lm, X, y, col, label):
    y_pred = lm.predict(X)
    r2 = lm.score(X, y[col])
    mse = mean_squared_error(y[col], y_pred)
    print(col + ':')
    print('R2 -' + label + 'set', r2)
    print('MSE -' + label + 'set', mse)

    return

def evaluate(norm_train, norm_test, target, reg):
    features = norm_train.columns
    targets = target.select_dtypes('number').columns


    X_train = norm_train.drop(targets, axis=1)
    y_train = norm_train[targets]
    X_test = norm_test.drop(targets, axis=1)
    y_test = norm_test[targets]
    y_col = y_test.columns

    for i in range(len(y_col)):
        # Create and fit the linear model
        # lm = Linear Model (variable name)
        lm = LinearRegression()

        # Fit to the train dataset
        lm.fit(X_train, y_train[y_col[i]])

        # alpha = intercept parameter (aka beta0)
        alpha = lm.intercept_

        # betas = coefficients
        betas = lm.coef_

        print('Intercept', alpha)
        print('Coefficients', betas)

        calc_r2mse(lm, X_test, y_test, y_col[i], 'test')
        calc_r2mse(lm, X_train, y_train, y_col[i], 'train')

        y_pred = lm.predict(X_test)
        plt.scatter(y_test[y_col[i]], y_pred, alpha=0.3)

        plt.title('Linear Regression (Predict Total)')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')

        plt.show()
        plt.savefig('evaluation ' + y_col[i][:3] + ' on ' + reg + '.png')

        residuals = y_test[y_col[i]] - y_pred

        # plot residuals
        plt.scatter(y_pred, residuals, alpha=0.3)

        # plot the 0 line (we want our residuals close to 0)
        plt.plot([min(y_pred), max(y_pred)], [0,0], color='red')

        plt.title('Residual Plot')
        plt.xlabel('Fitted')
        plt.ylabel('Residual')

        plt.show()
        plt.savefig('residual ' + y_col[i][:3] + ' on ' + reg + '.png')
        plt.clf()
        
    return

