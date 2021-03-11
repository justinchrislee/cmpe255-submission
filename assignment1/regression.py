import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

class Regression:
    def __init__(self):
        # get Boston housing data
        boston = load_boston()

        # split Boston data into multiple columns 
        self.df_x = pd.DataFrame(boston.data, columns=boston.feature_names)

        self.df_y = pd.DataFrame(boston.target)
    
    def linear_regression(self, features, finding_best_feature=False):
        reg = LinearRegression()

        # extract feature(s) from data
        if len(features) == 1:
            feature = self.df_x[features[0]].to_frame()
        else:
            feature = self.df_x[features]

        # split data into train and test data
        x_train, x_test, y_train, y_test = train_test_split(feature, self.df_y, test_size=0.3, random_state=4)

        # determine slope and y intercept for line
        reg.fit(x_train, y_train)

        # plug in test data inputs for line 
        predictions = reg.predict(x_test)

        # calculate rmse
        rmse = self.get_rmse(y_test, predictions)

        # if true return rmse to be used in find_best_feature function
        if finding_best_feature:
            return rmse
        else: 
            # calculate r squared
            r_squared = self.get_r_squared(y_test, predictions)

            # only plot data if exactly 1 feature is provided
            if len(features) == 1: 
                # plot data
                self.plot_data(x_test, y_test, predictions)

                # print linear regression scores
                print(f'Linear Regression RMSE: {rmse}')
                print(f'Linear Regression R Squared: {r_squared}')
            else:
                total_observations = len(y_test)
                independent_variable_totals = 3

                # calculate adjusted r squared
                adjusted_r_squared = self.get_adjusted_r_squared(r_squared, total_observations, independent_variable_totals)

                # print multiple regression scores
                print(f'Multiple Regression RMSE: {rmse}')
                print(f'Multiple Regression R Squared: {r_squared}')
                print(f'Multiple Regression Adjusted R Squared: {adjusted_r_squared}')
    
    def polynomial_regression(self, poly_reg_degree, selected_feature):
        # extract feature from df_x
        feature = self.df_x[selected_feature].to_frame()

        # split data into train and test data
        x_train, x_test, y_train, y_test = train_test_split(feature, self.df_y, test_size=0.3, random_state=4)

        poly = PolynomialFeatures(degree=poly_reg_degree)

        x_poly = poly.fit_transform(x_train)
        
        poly.fit(x_train, y_train)

        lin_reg = LinearRegression()
        lin_reg.fit(x_poly, y_train)

        predictions = lin_reg.predict(poly.fit_transform(x_test))

        # plot data
        self.plot_data(x_test, y_test, predictions)

        # calculate rmse
        rmse = self.get_rmse(y_test, predictions)

        # calculate r squared
        r_squared = self.get_r_squared(y_test, predictions)

        # print polynomial regression scores
        print(f'Polynomial Regression RMSE Degree {poly_reg_degree}: {rmse}')
        print(f'Polynomial Regression R Squared Degree {poly_reg_degree}: {r_squared}')
    
    def multiple_regression(self, features):
        self.linear_regression(features)

    def plot_data(self, x_test, y_test, predictions):
        plt.scatter(x_test, y_test, color='black')
        plt.plot(x_test, predictions, color='blue', linewidth=3)
        plt.show(block=True)
    
    def get_rmse(self, y_test, predictions):
        rmse = mean_squared_error(y_test, predictions, squared=False)
        return rmse
    
    def get_r_squared(self, y_test, predictions):
        sst_lam = lambda x: ((x - float(y_test.mean())) ** 2)
        sst = y_test.apply(sst_lam).sum()
        sse = ((predictions - y_test) ** 2).sum()
        r_squared = (1 - (sse / sst))[0]
        return r_squared
    
    def get_adjusted_r_squared(self, r_squared, n, p):
        adjusted_r_squared = 1-(1 - r_squared)*(n - 1)/(n-p-1)
        return adjusted_r_squared
    
    def find_best_feature(self):
        best_rmse_score = 0 
        best_feature = "CRIM" # default value prior to figuring out feature with best rmse

        # find feature with the best rmse
        for col in self.df_x.columns:
            rmse = self.linear_regression([col], True)

            if rmse > best_rmse_score:
                best_rmse_score = rmse
                best_feature = col
        
        return best_feature

if __name__ == "__main__":
    regression = Regression()
    best_feature = regression.find_best_feature() 
    regression.linear_regression([best_feature])
    regression.polynomial_regression(2, best_feature)
    regression.polynomial_regression(20, best_feature)
    regression.multiple_regression(['ZN', 'INDUS', 'RM'])
  
 