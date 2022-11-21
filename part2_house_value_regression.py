import torch as T
import torch.nn as nn
import torch.nn.functional as func
import pickle
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt


class NormMethod(Enum):
    MIN_MAX = 0
    Z_SCORE = 1


class Loss(Enum):
    MEAN_ABSOLUTE_ERROR = nn.L1Loss()
    MEAN_SQUARE_ERROR = nn.MSELoss()
    NEGATIVE_LOG_LIKELIHOOD = nn.NLLLoss()
    CROSS_ENTROPY = nn.CrossEntropyLoss()
    HINGE_EMBEDDING = nn.HingeEmbeddingLoss()
    MARGIN_RANKING = nn.MarginRankingLoss()
    TRIPLET_MARGIN = nn.TripletMarginLoss()
    KULLBACK_LEIBLER_DIVERGENCE = nn.KLDivLoss()


class Regressor:
    def __init__(
        self,
        x,
        nb_epoch=1000,
        learning_rate=0.001,
        normalisation_method=NormMethod.MIN_MAX,
        loss_function=Loss.MEAN_SQUARE_ERROR,
        batch_size=64,
        plot_loss=True,
    ):
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
        self.__learning_rate = learning_rate
        self.__normalisation_method = normalisation_method
        self.__loss_function = loss_function
        self.__device = "cuda" if T.cuda.is_available() else "cpu"
        self.__batch_size = batch_size
        self.__plot_loss = plot_loss

        self.__training_columns = None
        self.__x_mean = None
        self.__y_mean = None
        self.__x_std = None
        self.__y_std = None
        self.__x_min = None
        self.__y_min = None
        self.__x_max = None
        self.__y_max = None

        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Network Initialisation
        model = [
            # nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, self.output_size),
        ]

        for layer in model:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

        self.__network = nn.Sequential(*model)

        return


    def _preprocessor(self, x, y=None, training=False):
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

        if not training:
            if self.__training_columns is None:
                raise Exception("You need to train the model first!")
            if self.__normalisation_method == NormMethod.MIN_MAX:
                if self.__x_min is None or self.__x_max is None:
                    raise Exception("You need to train the model first!")
            if self.__normalisation_method == NormMethod.Z_SCORE:
                if self.__x_mean is None or self.__x_std is None:
                    raise Exception("You need to train the model first!")

        # One hot encoding of the discrete columns
        for feature_column in x:
            feature_values = x[feature_column]
            if feature_values.dtype == "object":
                new_columns = pd.get_dummies(feature_values)
                x = x.drop(feature_column, axis=1).join(new_columns)

        # Fill missing values with the mean of the column
        x = x.fillna(x.mean())

        # When training we initialise our normalisation values
        if training:
            self.__x_mean = x.mean()
            self.__x_std = x.std()
            self.__x_min = x.min()
            self.__x_max = x.max()
            if y is not None:
                self.__y_mean = y.mean()
                self.__y_std = y.std()
                self.__y_min = y.min()
                self.__y_max = y.max()


        # For Z-Score Normalisation
        if self.__normalisation_method == NormMethod.Z_SCORE:
                x = (x - self.__x_mean) / self.__x_std
                if y is not None:
                    y = (y - self.__y_mean) / self.__y_std
        # For Min-Max Normalisation
        elif self.__normalisation_method == NormMethod.MIN_MAX:
                x = (x - self.__x_min) / (self.__x_max - self.__x_min)
                if y is not None:
                    y = (y - self.__y_min) / (self.__y_max - self.__y_min)

        # Putting the columns correct for the test dataset, or setting them for the training
        if training:
            self.__training_columns = x.columns
        else:
            x = x.reindex(columns=self.__training_columns, fill_value=0)

        # Set X and Y types
        x = x.astype("float32")
        if y is not None:
            y = y.astype("float32")

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, pd.DataFrame) else None)

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

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        x_tensor = T.tensor(np.array(X))
        Y_Gold = T.tensor(np.array(Y))

        data_batch = T.utils.data.TensorDataset(x_tensor, Y_Gold)
        trainloader = T.utils.data.DataLoader(
            data_batch,
            batch_size=self.__batch_size,
            shuffle=True,
            num_workers=1,
        )

        permutation = T.randperm(X.shape[0])
        loss_by_epoch = []

        optimiser = T.optim.Adam(self.__network.parameters(), lr=self.__learning_rate)

        for i in range(self.nb_epoch):

            current_loss = 0

            for j, (data, data_y) in enumerate(trainloader):
                # Shuffle
                optimiser.zero_grad()

                # Forward
                Y_Pred = self.__network(data)

                """print(
                    "Data:\t",
                    int(data_y.mean().item()),
                    "\t\t",
                    "Prediction:\t",
                    int(Y_Pred.mean().item()),
                    "\t\t",
                    "Delta:\t",
                    int(data_y.mean().item()) - int(Y_Pred.mean().item()),
                )"""
                # Loss
                loss = self.__loss_function.value(Y_Pred, data_y)

                current_loss += loss.item() / self.__batch_size

                # Backward
                loss.backward()

                # Gradient Step
                optimiser.step()

                # Print statistics

            if (i + 1) % 1 == 0:
                print(f"Epoch {i+1}/{self.nb_epoch}, Train Loss: {current_loss:.4f}")

            loss_by_epoch.append(current_loss)

        if self.__plot_loss:
            plt.plot(range(self.nb_epoch), loss_by_epoch)
            plt.title("Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        x_tensor = T.tensor(np.array(X))
        return self.__network(x_tensor).detach().numpy()

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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        Y_Pred = self.__network(T.tensor(np.array(X)))
        Y_Gold = T.tensor(np.array(Y))

        return self.__loss_function.value(Y_Pred, Y_Gold).item()


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
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
    best_learning_rate = 0
    lowest_error = 10000000000

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    total_samples = len(data)
    train_fraction = 0.8
    si = train_fraction * total_samples

    x_train = data.loc[:si, data.columns != output_label]
    y_train = data.loc[:si, [output_label]]
    x_test = data.loc[si:, data.columns != output_label]
    y_test = data.loc[si:, [output_label]]

    for i in np.arange(0.01, 0.1, 0.01):
        regressor = Regressor(x_train, nb_epoch=10, learning_rate=i)
        regressor.fit(x_train, y_train)
        save_regressor(regressor)

        # Error
        error = regressor.score(x_train, y_train)
        print("\nTrain Regressor error: {}\n".format(error))

        error_test = regressor.score(x_test, y_test)
        print("\nTest Regressor error: {}\n".format(error_test))

        if error < lowest_error:
            best_learning_rate = i
            lowest_error = error

    print(best_learning_rate)


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    total_samples = len(data)
    train_fraction = 0.8
    si = train_fraction * total_samples

    # Splitting input and output
    x_train = data.loc[:si, data.columns != output_label]
    y_train = data.loc[:si, [output_label]]
    x_test = data.loc[si:, data.columns != output_label]
    y_test = data.loc[si:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=1)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nTrain Regressor error: {}\n".format(error))

    error_test = regressor.score(x_test, y_test)
    print("\nTest Regressor error: {}\n".format(error_test))

    RegressorHyperParameterSearch()



if __name__ == "__main__":
    example_main()
