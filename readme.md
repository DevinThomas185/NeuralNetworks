# Artificial Neural Networks Coursework

This repository contains:  
1. An implementation of a neural network [mini-library](part1_nn_lib.py).
2. A complete [regression neural network](part2_house_value_regression.py) trained on the "1990 California Housing" dataset.

This coursework was part of **COMP70050 - Introduction to Machine Learning (Autumn 2022)**, according to [specification](specifcation.pdf).

----

## Usage Guide for DoC Lab Machines
### Part 1 - Neural Network Mini-Library

In order to run part 1 of the coursework solution, invoke:
```
python3 part1_nn_lib.py
```  
Running the above, performs a simple test on the [mini-library](part1_nn_lib.py).   

In order to carry out further testing, a [test script](part1_tests.py) was written to ensure all layers of the mini-library functioned accordingly. This can be invoked by running:
```
python3 part_tests.py
```

----
 
### Part 2 - Trained Regression Neural Network

In order to run the main entrypoint of part 2 of the coursework solution, invoke:
```
python3 part2_house_value_regression.py
```  
This [regression neural network](part2_house_value_regression.py) is trained using training and validation sets. An evaluation using the test set is then performed.  

The regressor is initialised with the optimum parameters, obtained through hyperparameter tuning.  


Using **GridSearchCV** from [scikit-learn](https://scikit-learn.org/stable/install.html), an exhaustive search over specified parameter values was performed on the model.

### Flags
You will see these available flags:
```
Neural Networks and Evaluation Metrics

positional arguments:
  PATH_TO_DATASET       The relative path to the dataset
  OUTPUT_LABEL          The label for the output feature

optional arguments:
  -h, --help            show this help message and exit
  -cv, --validation     Use this flag to turn on cross validation
  -es, --early_stopping
                        Use this flag to enable early stopping
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to run the training for
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The rate at which learning is completed
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The size for each batch of mini-batched learning
  -d DROPOUT, --dropout DROPOUT
                        Provide a fraction of neurons to drop in training
  -n NEURONS [NEURONS ...], --neurons NEURONS [NEURONS ...]
                        Provide the hidden layers neurons
  -p, --plot_loss       Use this flag to plot the loss over time
  -x SEED, --seed SEED  Provide a seed for shuffling the dataset
  -s, --save            Save the resulting model to pickle file
```
### Example Use
You can run:
```
python3 part2_house_value_regression.py housing.csv median_house_value -cv -es -e 100 -lr 0.01 -b 100 -d 0.1  -n 5 5 -p -x 123 -s
```
Explanation:
- ```housing.csv``` run the regression neural network on **housing.csv** dataset
- ```median_house_value``` the label for output feature
- ```-cv``` enable cross-validation
- ```-es``` enable early stopping
- ```-e 100``` set epochs to 100
- ```-lr 0.01``` set learning rate to 0.01
- ```-b 100``` set batch-size to 100
- ```-d 0.1``` set dropout value to 0.1
- ```-n 5 5``` set neurons to [5, 5]
- ```-p``` enable plotting of loss over time
- ```-x 123``` set random shuffling seed to 123
- ```-s``` saves model as pickel file

----
### Part 2 - Hyperparameter Tuning
A external [script](hyperparameter_tuning.py) was written to fine tune and test hyperparameters:  
```
python3 hyperparameter_tuning.py housing.csv median_house_value
```
#### Flags
You will see these available flags:
```
Hyperparameter Tuning

positional arguments:
  PATH_TO_DATASET       The relative path to the dataset
  OUTPUT_LABEL          The label for the output feature

optional arguments:
  -h, --help            show this help message and exit
  -es, --early_stopping
                        Use this flag to enable early stopping
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to run the training for
  -lr LEARNING_RATE [LEARNING_RATE ...], --learning_rate LEARNING_RATE [LEARNING_RATE ...]
                        The rates at which learning is completed
  -b BATCH_SIZE [BATCH_SIZE ...], --batch_size BATCH_SIZE [BATCH_SIZE ...]
                        The sizes for each batch of mini-batched learning
  -d DROPOUT [DROPOUT ...], --dropout DROPOUT [DROPOUT ...]
                        Provide a list of fractions of neurons to drop in training
  -n NEURONS [NEURONS ...], --neurons NEURONS [NEURONS ...]
                        Provide the hidden layers neurons choices. Please separate choices with a , (e.g 5 5 : 10 10)
  -x SEED, --seed SEED  Provide a seed for shuffling the dataset
  -k FOLDS, --folds FOLDS
                        Provide the number of folds for cross validation
  -s, --save            Save the resulting model to pickle file
  ```
### Example Use
You can run:
```
python3 hyperparameter_tuning.py housing.csv median_house_value -es -e 10 -lr 0.01 0.1 -b 100 -d 0.1 0.9 -n 5 5 : 10 10
```
Explanation:
- ```housing.csv``` run the regression neural network on **housing.csv** dataset
- ```median_house_value``` the label for output feature
- ```-es``` enable early stopping
- ```-e 10``` set epochs to 10
- ```-lr 0.01 0.1``` set learning rate to [0.01, 0.1]
- ```-b 100``` set batch-size to 100
- ```-d 0.1``` set dropout value to 0.1
- ```-n 5 5: 10 10``` set neurons to [5, 5], [10, 10]  
  
----
### Extras
- [iris.dat](iris.dat) was used to test the neural network mini-library.
- [housing.csv](housing.csv) is a subdataset the "1990 California Housing" dataset, used to train the regression neural network.
- [part2_model.pickle](part2_model.pickle]) was created using the regression neural network, trained on the entire dataset with fine-tuned hyperparameters.
