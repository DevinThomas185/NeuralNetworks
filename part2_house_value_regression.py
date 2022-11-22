import torch as T
import torch.nn as nn
import torch.nn.functional as func
import pickle
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing, impute, metrics

# class NormMethod(Enum):
#     MIN_MAX = 0
#     Z_SCORE = 1

# class Loss(Enum):
#     MEAN_ABSOLUTE_ERROR = nn.L1Loss()
#     MEAN_SQUARE_ERROR = nn.MSELoss()
#     NEGATIVE_LOG_LIKELIHOOD = nn.NLLLoss()
#     CROSS_ENTROPY = nn.CrossEntropyLoss()
#     HINGE_EMBEDDING = nn.HingeEmbeddingLoss()
#     MARGIN_RANKING = nn.MarginRankingLoss()
#     TRIPLET_MARGIN = nn.TripletMarginLoss()
#     KULLBACK_LEIBLER_DIVERGENCE = nn.KLDivLoss()

class Regressor:
    def __init__(
        self,
        x,
        nb_epoch=1000,
        learning_rate=0.001,
        neurons=[8,8,8],
        loss_function=nn.MSELoss(),
        batch_size=1000,
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

        # Replace this code with your own
        self.__x = x
        self.__learning_rate = learning_rate
        self.__loss_function = loss_function
        # self.__device = T.device("cpu") 
        self.__device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        T.cuda.set_device(self.__device)
        self.__batch_size = batch_size
        self.__plot_loss = plot_loss

        self.__training_columns = None
        self.__label_replace = None
        self.__x_imputer = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
        self.__x_scaling = preprocessing.MinMaxScaler()
        self.__y_scaling = preprocessing.MinMaxScaler()

        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Network Initialisation
        self.__neurons = neurons
        model = []
        n_input = self.input_size
        for layer in neurons:
            model.append(nn.Linear(n_input, layer))
            n_input = layer
        model.append(nn.Linear(n_input, self.output_size))
        model.append(nn.ReLU())



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
            self.__label_replace = x['ocean_proximity'].mode()[0]
                        
        # Transform the ocean_proximity column into the one-hot encoded columns.
        x["ocean_proximity"] = x.loc[:, ['ocean_proximity']].fillna(value=self.__label_replace)
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
            if isinstance(y, pd.DataFrame):
                self.__y_scaling.fit(y)
    

        # Set X and Y types
        x = x.astype("float32")
        if y is not None:
            y = y.astype("float32")
        
        x = T.from_numpy(np.array(self.__x_scaling.transform(x))).to(self.__device)
        y = T.from_numpy(np.array(self.__y_scaling.transform(y))).to(self.__device) if isinstance(y, pd.DataFrame) else None
        

        return x, y

    def fit(self, x, y, dev_x=None, dev_y=None):
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
        data_batch = T.utils.data.TensorDataset(X, Y)
        trainloader = T.utils.data.DataLoader(
            data_batch,
            batch_size=self.__batch_size,
            shuffle=True,
        )

        permutation = T.randperm(X.shape[0])
        loss_by_epoch = []

        optimiser = T.optim.Adam(self.__network.parameters(), lr=self.__learning_rate)
        
        for i in range(self.nb_epoch):

            current_loss = 0

            for j, (data, data_y) in enumerate(trainloader):
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

                current_loss += loss.item()

                # Backward
                loss.backward()

                # Gradient Step
                optimiser.step()

                # Print statistics

            # print(f"Epoch {i+1}/{self.nb_epoch}, Train Loss: {current_loss:.4f}")

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

        X, Y = self._preprocessor(x, y=y, training=False) # Do not forget
        Y_Pred = self.__network(X)

        return self.__loss_function(Y_Pred, Y).item()



    def get_params(self, deep=True):
        return {
            'x': self.__x,
            'learning_rate': self.__learning_rate,
            'nb_epoch': self.nb_epoch,
            'neurons': self.__neurons,
            'batch_size': self.__batch_size
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

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


def RegressorHyperParameterSearch(params):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """
    from tune_sklearn import TuneGridSearchCV

    x, y = read_dataset("housing.csv", "median_house_value")
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                random_state=104, 
                                test_size=0.25, 
                                shuffle=True)

    # grid_search = TuneGridSearchCV(
    grid_search = GridSearchCV(
        Regressor(x=X_train, nb_epoch=10),
        param_grid=params,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        # use_gpu=True,
    )
    
    grid_search.fit(X_train, y_train, dev_x=X_test, dev_y=y_test)

    print("Best params:", grid_search.best_params_)

    # analysis = tune.run(train, config={"lr": tune.grid_search([0.001, 0.01, 0.1])}, resources_per_trial={"cpu": 28, "gpu": 1})
    # print("Best Config:", analysis.get_best_config(metric="mean_accuracy"))


def read_dataset(path, output_label):
    data = pd.read_csv(path)
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    return x, y


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
    print("Train Regressor error: {}".format(error))

    error_test = regressor.score(x_test, y_test)
    print("Test Regressor error: {}".format(error_test))


    params = {
        'learning_rate': [0.001, 0.005, 0.05, 0.1],
        'batch_size': [64, 256, 512, 1024],
        'neurons': [[13], [10, 10], [8, 8], [8, 8, 8], [13, 8], [9, 4], [13, 9, 4]]
    }

    RegressorHyperParameterSearch(params)



if __name__ == "__main__":
    example_main()
