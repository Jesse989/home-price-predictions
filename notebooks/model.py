from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

import statsmodels.api as sm
import pandas as pd
import numpy as np

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=False):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# adjusted R-squared:
def adj_r2(r2_score, num_observations, num_parameters):
    """Calculate the Adjusted R-Squared value

    Args: 
        r2_score (int): R-Squared value to adjust
        num_observations (int): Number of observations used in model
        num_parameters (int): Number of parameters used in model

    Returns:
        adj_r2 (float): Adjusted R-Squared value
    """
    return r2_score-(num_parameters-1)/(num_observations-num_parameters)*(1-r2_score)


class Scores:
    '''Keeps track and displays model metrics'''

    def __init__(self):
        self.score_index = -1
        self.scores = pd.DataFrame({'Description': [],
                                    'RMSE (training, test)': [],
                                    'R-squared (training, test)': [],
                                    'Adjusted R-squared (training, test)': [],
                                    '5-Fold Cross Validation': []})
        self.coefs = pd.DataFrame()
        self.sm_results = []

    def evaluate_model(self, description, data, encode_dummies=[], select_features=False):
        """ Creates model and saves results
        
        Args: 
            description (str): Description of data to evaluate
            data (DataFrame): pandas DataFrame
            
        """
        
        # encode one hot if passed in
        if len(encode_dummies) > 0:
            data = pd.get_dummies(
                data, prefix=encode_dummies, columns=encode_dummies, drop_first=True)
                    
        # run stepwise_selection to determine relevant features
        # is set to True
        if select_features:
            relevant_columns = stepwise_selection(data.drop('price', axis=1), data['price'])
        else:
            relevant_columns = list(data.columns)
            relevant_columns.remove('price')
        
        # split data into X and y
        X = data[relevant_columns].copy()
        y = data['price'].copy()
                
        # split data into testing set and training set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)
    
        # create linear regression model
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        
        # create ols model for summary stats
        sm_model = sm.OLS(y, sm.add_constant(X))
        self.sm_results.append(sm_model.fit())
        
        # index the results
        self.score_index += 1
        df_idx = self.score_index
        
        # track model coefs and store them in dataframe
        coefs = pd.DataFrame(np.reshape(
            linreg.coef_, (1, len(X.columns))), columns=X.columns, index=[df_idx])
        self.coefs = pd.concat([self.coefs, coefs], axis='rows', sort=True)
        
        # predict training set, and testing set
        y_hat_train = linreg.predict(X_train)
        y_hat_test = linreg.predict(X_test)
        
        # calculate RMSE and save to results DataFrame
        train_rmse = np.sqrt(mean_squared_error(y_train, y_hat_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_hat_test))
        
        # calcuate r-squared and adjusted r-squared
        train_r_squared = r2_score(y_train, y_hat_train)
        train_r_squared_a = adj_r2(
            train_r_squared, len(y_train), len(linreg.coef_))

        test_r_squared = r2_score(y_test, y_hat_test)
        test_r_squared_a = adj_r2(
            test_r_squared, len(y_test), len(linreg.coef_))

        # calculate cross validation score
        k_fold = np.mean(cross_val_score(linreg, X, y, cv=5,
                                         scoring='r2'))

        # create a result with all the information
        result = {'Description': [description],
                  'RMSE (training, test)': [(format(train_rmse, '.3f'), format(test_rmse, '.3f'))],
                  'R-squared (training, test)': [(format(train_r_squared, '.3f'), format(test_r_squared, '.3f'))],
                  'Adjusted R-squared (training, test)': [(format(train_r_squared_a, '.3f'), format(test_r_squared_a, '.3f'))],
                  '5-Fold Cross Validation': [format(k_fold, '.3f')]}
        
        # turn result into a DataFrame
        result = pd.DataFrame(result, index=[df_idx])
        
        # combine result with previous results
        self.scores = pd.concat([self.scores, result], axis='rows')
        
        return self.scores
    
    def plot_feature(self, feature, idx=-1, fig=None):
        """Plot the model given the scores index number
        
        args: 
            feature (string): feature to plot
            idx (int): index of score in .scores
            fig (matplotlib figure): fig to pass to sm.graphics.plot
        returns:
            fig: plot using sm.graphics.plot_regress_exog()"""
        
        result = self.sm_results[idx]
        return sm.graphics.plot_regress_exog(result, feature, fig=fig)
    
    def plot_qq(self, idx=-1, fig=None):
        """Plot the qq for given index number
        
        args: 
            idx (int): index of score in .scores
        returns:
            fig: plot using sm.graphics.qqplot()"""
        result = self.sm_results[idx]
        return sm.graphics.qqplot(result.resid, dist=stats.norm, line='45', fit=True)