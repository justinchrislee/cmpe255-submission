import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

class Regression:
    def __init__(self):
        df = pd.read_csv(filepath_or_buffer='housing.csv', sep="\s+",
        names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B1000', 'LSTAT', 'MEDV'])
        
        self.df_x = df.loc[:, 'CRIM':'LSTAT']
        self.df_y = df.loc[:, 'MEDV':'MEDV']
    
    def linear_regression(self, features, finding_best_feature=False):
        reg = LinearRegression()

        # extract feature(s) from data
        if len(features) == 1:
            feature = self.df_x[features[0]].to_frame()
        else:
            feature = self.df_x[features]

        # split data into train and test data
        x_train, x_test, y_train, y_test = train_test_split(feature, self.df_y, test_size=0.3, random_state=42)

        # determine slope and y intercept for line
        reg.fit(x_train, y_train)

        # plug in test data inputs for line 
        predictions = reg.predict(x_test)

        # calculate r squared
        r_squared = self.get_r_squared(y_test, predictions)

        # if true return rmse to be used in find_best_feature function
        if finding_best_feature:
            return r_squared
        else: 
            # calculate rmse
            rmse = self.get_rmse(y_test, predictions)

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
        # extract feature from x_y
        feature = self.df_x[selected_feature].to_frame()

        # split data into train and test data
        x_train, x_test, y_train, y_test = train_test_split(feature, self.df_y, test_size=0.3, random_state=42)
        
        model = Pipeline([('poly', PolynomialFeatures(degree=poly_reg_degree)), ('linear', LinearRegression())])

        model = model.fit(x_train, y_train)
        
        # sort data from least to greatest for plotting
        x_y_test_data_combined = pd.concat([x_test, y_test], axis=1)
        x_y_test_sorted = x_y_test_data_combined.sort_values(by=selected_feature)
        
        x_test = x_y_test_sorted[[selected_feature]]
        y_test = x_y_test_sorted[['MEDV']]

        predictions = model.predict(x_test)

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
        y_test = pd.DataFrame(y_test)
        error = predictions - y_test
        mse = (error ** 2).mean()
        rmse = float(np.sqrt(mse))
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
        best_r_squared_score = 0 
        best_feature = "CRIM" # default value prior to figuring out feature with best rmse

        # find feature with the best r_squared value
        for col in self.df_x.columns:
            r_squared = self.linear_regression([col], True)
            if r_squared > best_r_squared_score:
                best_r_squared_score = r_squared
                best_feature = col
        
        return best_feature

if __name__ == "__main__":
    regression = Regression()
    best_feature = regression.find_best_feature() 
    regression.linear_regression([best_feature])
    regression.polynomial_regression(2, best_feature)
    regression.polynomial_regression(20, best_feature)
    regression.multiple_regression(['ZN', 'INDUS', 'RM'])
  
 