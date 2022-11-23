import torch as T
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, impute
import sys
from argparse import ArgumentParser

DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NEURONS = [5, 5]
DEFAULT_BATCH_SIZE = 100


class Regressor:
    def __init__(
        self,
        x,
        validation=False,
        early_stop=False,
        dropout=None,
        nb_epoch=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        neurons=DEFAULT_NEURONS,
        batch_size=DEFAULT_BATCH_SIZE,
        plot_loss=False,
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
        # Hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.neurons = neurons
        self.early_stop = early_stop
        self.dropout = dropout

        self.__x = x
        self.__loss_function = nn.MSELoss()
        self.__validation = validation

        # Enable GPU usage if available
        device_name = "cuda:0" if T.cuda.is_available() else "cpu"
        self.__device = T.device(device_name)

        if device_name != "cpu":
            T.cuda.set_device(self.__device)

        # Plotting boolean
        self.__plot_loss = plot_loss

        # Preprocessing specific values
        self.__training_columns = None
        self.__label_replace = None
        self.__x_imputer = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
        self.__x_scaling = preprocessing.MinMaxScaler()

        # Initialise preprocesor values
        X, _ = self._preprocessor(x, training=True)

        # Network Initialisation
        self.input_size = X.shape[1]
        self.output_size = 1

        model = []
        n_input = self.input_size
        for layer in neurons:
            model.append(nn.Linear(n_input, layer))
            if self.dropout is not None:
                model.append(nn.Dropout(self.dropout))
            n_input = layer
        model.append(nn.Linear(n_input, self.output_size))
        model.append(nn.ReLU())  # Ensure that no negative house prices are predicted

        for layer in model:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

        self.__network = nn.Sequential(*model).to(self.__device)
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

        if training:
            self.__label_replace = x["ocean_proximity"].mode()[0]

        # Transform the ocean_proximity column into the one-hot encoded columns.
        pd.options.mode.chained_assignment = None  # Disable pandas unnecessary warning
        x["ocean_proximity"] = x.loc[:, ["ocean_proximity"]].fillna(
            value=self.__label_replace
        )
        new_columns = pd.get_dummies(x["ocean_proximity"])
        x = x.drop("ocean_proximity", axis=1).join(new_columns)

        # Putting the columns correct for the test dataset, or setting them for the training
        if training:
            self.__training_columns = x.columns
        else:
            x = x.reindex(columns=self.__training_columns, fill_value=0)

        if training:
            self.__x_imputer.fit(x)

        x = self.__x_imputer.transform(x)
        if training:
            self.__x_scaling.fit(x)

        # Set X and Y types
        x = x.astype("float32")
        if y is not None:
            y = y.astype("float32")

        x = T.from_numpy(np.array(self.__x_scaling.transform(x))).to(self.__device)
        y = (
            T.from_numpy(np.array(y)).to(self.__device)
            if isinstance(y, pd.DataFrame)
            else None
        )

        return x, y

    def fit(self, x, y, validation_x=None, validation_y=None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        if (validation_x is None or validation_y is None) and self.__validation:
            print("No validation dataset provided, switching to no validation")
            self.__validation = False

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        data_batch = T.utils.data.TensorDataset(X, Y)
        trainloader = T.utils.data.DataLoader(
            data_batch,
            batch_size=self.batch_size,
            shuffle=True,
        )

        loss_by_epoch = []
        val_loss_by_epoch = []

        optimiser = T.optim.Adam(self.__network.parameters(), lr=self.learning_rate)

        min_val_loss = 0

        for i in range(self.nb_epoch):

            running_loss = 0
            last_loss = 0

            for data, data_y in trainloader:
                data = data.to(self.__device)
                data_y = data_y.to(self.__device)

                # Shuffle
                optimiser.zero_grad()

                # Forward
                Y_Pred = self.__network(data)

                """print(
                    "Batch:\t",
                    j,
                    "\t\t",
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
                loss = self.__loss_function(Y_Pred, data_y)

                running_loss += loss.item()

                # Backward
                loss.backward()

                # Gradient Step
                optimiser.step()

            last_loss = running_loss / len(trainloader)

            # Early stop depending on if validation loss increases
            if self.__validation:
                validation_loss = self.score(validation_x, validation_y)
                val_loss_by_epoch.append(validation_loss)
                # Initialise min_val_loss properly
                if i == 0:
                    min_val_loss = validation_loss

            if self.__plot_loss:
                if self.__validation:
                    print(
                        f"Epoch {i+1}/{self.nb_epoch}, Train Loss: {last_loss:.4f}, Validation Loss: {validation_loss:.4f}"
                    )
                else:
                    print(f"Epoch {i+1}/{self.nb_epoch}, Train Loss: {last_loss:.4f}")

            loss_by_epoch.append(last_loss)

            if self.__validation:
                if validation_loss <= min_val_loss:
                    min_val_loss = validation_loss
                else:
                    if self.early_stop:
                        break

        self.x_axis = range(1, i + 2)
        self.y_axis = loss_by_epoch

        if self.__plot_loss:
            plt.plot(self.x_axis, self.y_axis, label="Training Loss")
            if self.__validation:
                plt.plot(self.x_axis, val_loss_by_epoch, label="Validation Loss")
            plt.title("Loss by Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.vlines(
                get_elbow_value(self), plt.ylim()[0], plt.ylim()[1], linestyles="dashed"
            )
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
        return self.__network(X).cpu().detach().numpy()

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
        Y_Pred = self.__network(X)

        return self.__loss_function(Y_Pred, Y).item()

    def get_params(self, deep=True):
        return {
            "x": self.__x,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "early_stop": self.early_stop,
            "learning_rate": self.learning_rate,
            "neurons": self.neurons,
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self


def get_elbow_value(model):
    # Need to account for if someone does not have kneed, then we return dummy 0
    try:
        from kneed import KneeLocator

        return KneeLocator(
            model.x_axis, model.y_axis, S=1.0, curve="convex", direction="decreasing"
        ).knee
    except ImportError:
        return -1


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


def RegressorHyperParameterSearch(
    seed,
    save,
    folds,
    epochs,
    path_to_dataset,
    output_label,
    params,
):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """
    x, y = read_dataset(path_to_dataset, output_label)

    X_VT, X_test, y_VT, y_test = train_test_split(
        x, y, random_state=seed, test_size=1 / folds, shuffle=True
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_VT, y_VT, random_state=seed, test_size=1 / (folds - 1), shuffle=True
    )

    grid_search = GridSearchCV(
        Regressor(x=X_train, validation=True, nb_epoch=epochs),
        param_grid=params,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        cv=folds,
        return_train_score=True,
    )

    grid_search.fit(
        X_train, y_train, validation_x=X_validation, validation_y=y_validation
    )

    print("Best params:", grid_search.best_params_)
    print("Elbow Epoch:", get_elbow_value(grid_search.best_estimator_))

    error_test = math.sqrt(grid_search.best_estimator_.score(X_test, y_test))
    print("Test Regressor error: {}".format(error_test))

    if save:
        save_regressor(grid_search.best_estimator_)


def read_dataset(path, output_label):
    data = pd.read_csv(path)
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    return x, y


def _run_neural_net(
    path_to_dataset,
    output_label,
    validation,
    early_stop,
    nb_epoch,
    learning_rate,
    batch_size,
    dropout,
    neurons,
    plot_loss,
    seed,
    save,
):

    x, y = read_dataset(path_to_dataset, output_label)

    X_VT, X_test, y_VT, y_test = train_test_split(
        x, y, random_state=seed, test_size=1 / 10, shuffle=True
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_VT, y_VT, random_state=seed, test_size=1 / 9, shuffle=True
    )

    training_set = X_train if validation else X_VT

    regressor = Regressor(
        training_set,
        plot_loss=plot_loss,
        validation=validation,
        early_stop=early_stop,
        nb_epoch=nb_epoch,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout=dropout,
        neurons=neurons,
    )

    if validation:
        regressor.fit(
            X_train, y_train, validation_x=X_validation, validation_y=y_validation
        )
    else:
        regressor.fit(X_VT, y_VT)

    if save:
        save_regressor(regressor)

    print("Optimal Epochs:", get_elbow_value(regressor))

    # Error
    error_test = math.sqrt(regressor.score(X_test, y_test))
    print("Test Regressor error: {}".format(error_test))

    # params = {
    #     'batch_size': [100, 1000],
    #     'dropout': [None, 0.1, 0.5, 0.8],
    #     'early_stop': [True, False],
    #     'learning_rate': [0.01, 0.1],
    #     'neurons': [[10], [13], [5, 5]],
    # }

    # RegressorHyperParameterSearch(params)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Neural Networks", description="Neural Networks and Evaluation Metrics"
    )

    parser.add_argument(
        "PATH_TO_DATASET",
        help="The relative path to the dataset",
    )

    parser.add_argument(
        "OUTPUT_LABEL",
        help="The label for the output feature",
    )

    parser.add_argument(
        "-cv",
        "--validation",
        action="store_true",
        help="Use this flag to turn on cross validation",
    )

    parser.add_argument(
        "-es",
        "--early_stopping",
        action="store_true",
        help="Use this flag to enable early stopping",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="The number of epochs to run the training for",
        default=DEFAULT_EPOCHS,
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The rate at which learning is completed",
        default=DEFAULT_LEARNING_RATE,
        type=float,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        help="The size for each batch of mini-batched learning",
        default=DEFAULT_BATCH_SIZE,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--dropout",
        help="Provide a fraction of neurons to drop in training",
        default=None,
        type=float,
    )

    parser.add_argument(
        "-n",
        "--neurons",
        nargs="+",
        help="Provide the hidden layers neurons",
        default=DEFAULT_NEURONS,
        type=int,
    )

    parser.add_argument(
        "-p",
        "--plot_loss",
        action="store_true",
        help="Use this flag to plot the loss over time",
    )

    parser.add_argument(
        "-x",
        "--seed",
        help="Provide a seed for shuffling the dataset",
        default=None,
        type=int,
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the resulting model to pickle file",
    )

    args = parser.parse_args(sys.argv[1:])

    _run_neural_net(
        path_to_dataset=args.PATH_TO_DATASET,
        output_label=args.OUTPUT_LABEL,
        validation=args.validation,
        early_stop=args.early_stopping,
        nb_epoch=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dropout=args.dropout,
        neurons=list(args.neurons),
        plot_loss=args.plot_loss,
        seed=args.seed,
        save=args.save,
    )
