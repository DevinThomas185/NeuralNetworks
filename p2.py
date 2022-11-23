import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute, metrics, model_selection
from sklearn.base import BaseEstimator

import math
import copy
import sys

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class Regressor():

    @staticmethod
    def _init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)

    def __init__(self, x, nb_epoch = 1000, neurons = [8, 8, 8], learning_rate = 0.001, loss_fun = "mse", batch_size = 64):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Values stored for pre-processing
        self.x = x
        self.x_scaler = preprocessing.MinMaxScaler() # Perfoms min-max scaling on x values
        self.y_scaler = preprocessing.MinMaxScaler() # Performs min-max scaling on y values
        self.x_imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean') # Used to handle empty cells
        self.lb = preprocessing.LabelBinarizer() # Used to handle ocean_proximity
        self.lb.classes_ = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'] # Hard code the class labels
        self.string_imp = None # Used to handle empty ocean_proximities

        self.__device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        T.cuda.set_device(self.__device)

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Initialising Net stuff
        self.neurons = neurons # Architecture of net
        layers = []
        n_in = self.input_size
        for layer in neurons:
            layers.append(nn.Linear(n_in, layer)) # Use Linear activation functions only
            n_in = layer
        layers.append(nn.Linear(n_in, self.output_size))


        layers.append(nn.ReLU()) # Use ReLU as final activation function
        
        self.net = nn.Sequential(*layers) # Stack-Overflow Bless
        self.net.apply(self._init_weights)
        self.net.double()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stop = 20
        self.best_iteration = -1
        if loss_fun == "mse":
            self.loss_layer = nn.MSELoss()
        else:
            raise Exception(f'Undefined loss_fun: {loss_fun}')
        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        if training: self.x = x # Need to store this for GridSearchCV (dumb)

        # First we handle the strings
        # Deal with empty cells in string
        if training: self.string_imp = x['ocean_proximity'].mode()[0]

        pd.options.mode.chained_assignment = None # Suppress warning (it's wrong)
        x['ocean_proximity'] = x.loc[:, ['ocean_proximity']].fillna(value=self.string_imp)

        # Replace strings with binary values
        proximity = self.lb.transform(x['ocean_proximity'])
        x = x.drop('ocean_proximity', axis=1)
        x =x.join(pd.DataFrame(proximity))

        # Next we impute (deal with empty cells)
        if training: self.x_imp.fit(x)

        x = self.x_imp.transform(x)
        # If training we initialise our normalisation values
        if training:
            self.x_scaler.fit(x)
            if isinstance(y, pd.DataFrame): self.y_scaler.fit(y)

        x = torch.from_numpy(self.x_scaler.transform(x)).to(self.__device)
        y = torch.from_numpy(self.y_scaler.transform(y)).to(self.__device) if isinstance(y, pd.DataFrame) else None
        
        return x, y


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, dev_x = None, dev_y = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Tracks parameters for early stopping
        min_loss = float("inf")
        best = self

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        X_size = X.size()[0]
        batch_size = min(self.batch_size, X_size) # Make sure batches aren't too big

        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

        for i in range(self.nb_epoch):
            running_loss = []

            # Use random batches each epoch
            permutation = torch.randperm(X_size)

            for j in range(0, X_size, batch_size):
                optimizer.zero_grad()

                # Select batch
                indices = permutation[j:j+batch_size]
                batch_X, batch_Y = X[indices], Y[indices]

                # Forward + Backward Pass
                output = self.net(batch_X)
                loss = self.loss_layer(output, batch_Y)
                loss.backward()
                optimizer.step()
                
                # Calculate RMSE for batch
                y_hat = self.y_scaler.inverse_transform(output.detach().numpy())
                y_gold = y.to_numpy()[indices]
                score = metrics.mean_squared_error(y_gold, y_hat, squared=False)
                running_loss.append(score)
                
            # Calculate RMSE for epoch
            scaled_loss = sum(running_loss)/len(running_loss)

            #print(f'Loss at epoch {i}: {scaled_loss}')

            # Using validation set for early stopping
            if dev_x is not None and dev_y is not None:
                validation_loss = self.score(dev_x, dev_y)

                if validation_loss < min_loss:
                    self.best_iteration = i
                    best = copy.deepcopy(self)
                    min_loss = validation_loss
                else:
                    if i - self.best_iteration > self.early_stop:
                        return best

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        output = self.net(X).detach().numpy()

        return self.y_scaler.inverse_transform(output)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        output = self.net(X).detach().numpy()

        y_hat = self.y_scaler.inverse_transform(output)
        y_gold = y.to_numpy()
        return metrics.mean_squared_error(y_gold, y_hat, squared=False)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def get_params(self, deep=True):
        return {
            'x': self.x,
            'learning_rate': self.learning_rate,
            'nb_epoch': self.nb_epoch,
            'neurons': self.neurons,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        
        return self


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y, params): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
        - params {dictionary} -- Dictionary with parameter names (str) as keys and lists of parameter settings to try as values
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    test = 8
    dev = 1
    train = 1

    x_size = len(x.index)
    fold_size = x_size // (test + dev + train)

    permutation = torch.randperm(x_size)
    test_split = permutation[:fold_size * test]
    dev_split = permutation[fold_size * test:fold_size * (test + dev)]
    train_split = permutation[fold_size * (test + dev):]

    x_train = x.iloc[train_split]
    y_train = y.iloc[train_split]

    x_dev = x.iloc[dev_split]
    y_dev = y.iloc[dev_split]

    gs = model_selection.GridSearchCV(
        Regressor(x=x_train), 
        param_grid=params, 
        n_jobs=-1, # Set n_jobs to -1 for parallelisation
        scoring='neg_root_mean_squared_error',
        verbose=2, 
        return_train_score=True)

    gs.fit(x, y, dev_x=x_dev, dev_y=y_dev)

    original_stdout = sys.stdout
    filename = "results.txt"
    res = pd.DataFrame(gs.cv_results_)
    print(f"Saving results to {filename}")

    with open(filename, "w") as outfile:
        sys.stdout = outfile
        print(res[['param_batch_size', 'param_learning_rate', 'param_neurons', 'mean_test_score', 'std_test_score', 'mean_train_score']])
        sys.stdout = original_stdout

    print("Grid scores on dev set:")
    print(gs.best_score_)
    print("Best learning rate:", gs.best_estimator_.learning_rate)
    print("Best neuron layout:", gs.best_estimator_.neurons)
    print("Stopping epoch:", gs.best_estimator_.best_iteration)
    #save_regressor(gs.best_estimator_)
    
    return  gs.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():
    import time
    s = time.time()

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    test = 8
    dev = 1
    train = 1

    x_size = len(x.index)
    fold_size = x_size // (test + dev + train)

    permutation = torch.randperm(x_size)
    test_split = permutation[:fold_size * test]
    dev_split = permutation[fold_size * test:fold_size * (test + dev)]
    train_split = permutation[fold_size * (test + dev):]

    x_train = x.iloc[train_split]
    y_train = y.iloc[train_split]

    x_dev = x.iloc[dev_split]
    y_dev = y.iloc[dev_split]

    x_test = x.iloc[test_split]
    y_test = y.iloc[test_split]
        

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 272, neurons = [12, 8], learning_rate = 0.001, batch_size = 512)
    regressor.fit(x_train, y_train, x_dev, y_dev)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # Test predict
    result = regressor.predict(x_test)
    print(result)

    # Print stopping time
    print("Stopping time:", regressor.best_iteration)

    print(time.time() - s)

if __name__ == "__main__":
    example_main()
