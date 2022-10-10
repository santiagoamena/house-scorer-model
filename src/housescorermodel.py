"""
House Scorer Model

This script takes the houses prices dataset provided and fits a
model to predict sale prices

The input for this tool must be the provided csv file 
"challenge_houses-prices.csv"

This script requires that `pandas`, `xgboost` and `scikit-learn` be 
installed within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * fit - returns the column headers of the file
    * get_model_performance - returns the training and test mae and rmae
    * get_prediction - predicts a sale price for a single observation
"""

# Author: Santiago Amena <santiagoamena@gmail.com>

import numpy as np
import pandas as pd 

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class HouseScorerModel:
    """
    A class used to fit a gradient boosting model to the 
    challenge_houses-prices dataset and get predictions 

    ...

    Attributes
    ----------
    df : pandas DataFrame
        the data in the csv file read as a pandas dataframe
    xgbmodel : xgboost regressor
        the regression model defined
    X_train : str
        the training set predictors
    y_train : str
        the training set variable to predict
    X_test : str
        the testing set predictors
    y_test : str
        the testing set variable to predict 

    Methods
    -------
    fit()
        Fits the model

    get_model_performance()
        Gets the results obtained in the fit() method

    get_prediction(input_value)
        Gets single prediction taking a single dict-like input value
    """

    def __init__(self, datafile = "challenge_houses-prices.csv"):
        """
        Parameters
        ----------
        datafile : str, path object or file-like object
            The path to the "challenge_houses-prices" file
        """
        # Read the data
        self.df = pd.read_csv(datafile)

        # Do some preprocessing
        df_house_style = pd.get_dummies(self.df['house_style'])
        df_neighborhood = pd.get_dummies(self.df['neighborhood'])
        self.df = pd.concat([self.df, df_house_style, df_neighborhood], axis=1)
        self.df.drop(['house_style','neighborhood'], axis=1, inplace=True)
        for column in self.df.columns:
            self.df[column] = self.df[column].astype(float)

        # Define model with parameters obtained in bayesian optimization
        self.xgbmodel = XGBRegressor(
            n_estimators=100,
            objective='reg:squarederror',
            eval_metric=['rmse', 'mae'],
            booster='gbtree',
            eta=0.09192742232542225,
            gamma=1.7659450508306334,
            max_depth=6,
            min_child_weight=6.997838154971065,
            max_delta_step=0,
            subsample=0.13915515897442735,
            colsample_bytree=0.8832635679721044,
            reg_alpha=70.27689014791974,
            reg_lambda=41.17989096578065,
            max_leaves=932,
            max_bin=484,
            n_jobs=-1,
            random_state=1,
        )

        # Define training and testing sets
        y = self.df['sale_price']
        X = self.df.drop('sale_price', axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )

    def fit(self):
        """
        Takes the training set and fits the model, then gets the results 
        on the test set and gets feature importances.

        """
        # Fit the model
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model = self.xgbmodel.fit(
            self.X_train, 
            self.y_train,
            eval_set=eval_set,
            verbose=0    
        )

        # Get results (mean absolute error)
        results = self.model.evals_result()
        self.train_mae = results['validation_0']['mae']
        self.test_mae = results['validation_1']['mae']
        self.train_rmae = np.sqrt(self.train_mae)
        self.test_rmae = np.sqrt(self.test_mae)

        # Get feature importances
        sorted_idx = (-self.model.feature_importances_).argsort()
        cols = np.array(self.X_train.columns)[sorted_idx]
        imp_values = self.model.feature_importances_[sorted_idx]
        self.importance = dict(zip(cols, imp_values))


    def get_model_performance(self):
        """
        Returns the model performance as measured by the mean absolute error,
        both on the train and test sets.

        """
        self.results = {
            "train": {"rmae": self.train_rmae[-1], "mae": self.train_mae[-1]},
            "test": {"rmae": self.test_rmae[-1], "mae": self.test_mae[-1]},
        }
        return self.results

    def get_prediction(self, input_value):
        """
        Returns a sale price prediction for a single input value, as well
        as the top features of the model.

        Parameters
        ----------
        input_value : dict
            A dict containing the values for all the variables in the dataset
            for a single observation.
        """
        # Convert dict input to pandas dataframe
        temp_df = pd.DataFrame(input_value, index=[0])

        # Process data
        temp_df_house_style = pd.get_dummies(temp_df['house_style'])
        temp_df_neighborhood = pd.get_dummies(temp_df['neighborhood'])
        temp_df = pd.concat(
            [temp_df, temp_df_house_style, temp_df_neighborhood], axis=1
        )
        temp_df.drop(['house_style','neighborhood'], axis=1, inplace=True)
        for column in temp_df.columns:
            temp_df[column] = temp_df[column].astype(float)
        cols = self.X_train.columns.to_list()
        df1 = pd.DataFrame(columns=cols)
        df1 = pd.concat([df1, temp_df], axis=0)
        df1.fillna(0, inplace=True)

        # Get prediction
        self.y_pred = self.xgbmodel.predict(df1)
        self.pred_dict = {
            "prediction": self.y_pred[0],
            "top_features": self.importance
        }
        return self.pred_dict

