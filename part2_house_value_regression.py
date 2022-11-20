import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        #X, _ = self._preprocessor(x, training = True)
        self._preprocessor(x, training = True)
        #self.input_size = X.shape[1]
        #self.output_size = 1
        #self.nb_epoch = nb_epoch
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
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Steps according to the spec:

        # Store parameters used for preprocessing to be able to apply the same
        # process to all inputs of the model

        # Make sure that the training boolean flag is taken into account.
        # If it is training, it should calculate the new global preprocessing values
        # (I assume those are the mean and standard deviation for the standardisation
        # of input) and if training is false, we should use the previously computed
        # values. IMPORTANT

        # Handle the missing values in the data using Pandas fillna function.
        # Idea: since all values have some real-live meaning, I was thinking about
        # replacing missing values with the average of all values for that feature.

        # Handle the textual values in the data using Sklearn LabelBinarizer

        # Normalise the numerical values to improve learning.

        #######################################################################

        x_numerical = x.loc[:, x.columns != "ocean_proximity"]

        if training:
            #Recompute the processing values if training
            self.numerical_means = x_numerical.mean()
            # Adding a sane default for ocean proximity
            self.standard_deviations = x_numerical.std()
            # We also want to fit the encoder only when training
            self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")
            self.one_hot_encoder.fit(x[["ocean_proximity"]])


        # Fills in all NaNs with our previously computed means.
        fill_mask = self.numerical_means.copy()
        # Fill in a sane default for missing ocean_proximity entries."
        fill_mask["ocean_proximity"] = "INLAND"
        x = x.fillna(value=fill_mask)


        # Transform the ocean_proximity column into the one-hot encoded columns.
        one_hot_dataframe = pd.DataFrame(self.one_hot_encoder.transform(x[["ocean_proximity"]]).toarray())

        # Normalise the data using the previously computed means.
        x_normalised = (x_numerical - self.numerical_means)/self.standard_deviations


        # Append the one-hot encoded columns for the ocean_proximity to the end.
        x = x_normalised.join(one_hot_dataframe)

        X = torch.Tensor(np.array(x))

        # Do we also want to pre-process y somehow?
        Y = (torch.tensor(y) if isinstance(y, pd.DataFrame) else None)
        print(X)
        print(Y)
        return X, Y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def fit(self, x, y):
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

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


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



def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    #regressor.fit(x_train, y_train)
    #save_regressor(regressor)

    # Error
    #error = regressor.score(x_train, y_train)
    #print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

