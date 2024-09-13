import statsmodels.api as sm
import pandas as pd

#read in data
data = pd.read_csv('train.csv')

# define variables
x = data['x'].tolist()
y = data['y'].tolist()

# adding the constant term

x = sm.add_constant(x)

# do regression and fit model
result = sm.OLS(y, x).fit()

#print summary table
print(result.summary())

#   Description of some of the terms in the table :
#   R-squared : the coefficient of determination. It is the proportion of the variance in the dependent variable that is predictable/explained
#   Adj. R-squared : Adjusted R-squared is the modified form of R-squared adjusted for the number of independent variables in the model. Value of adj. R-squared increases, when we include extra variables which actually improve the model.
#   F-statistic : the ratio of mean squared error of the model to the mean squared error of residuals. It determines the overall significance of the model.
#   coef : the coefficients of the independent variables and the constant term in the equation.
#   t : the value of t-statistic. It is the ratio of the difference between the estimated and hypothesized value of a parameter, to the standard error
