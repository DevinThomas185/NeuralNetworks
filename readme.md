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
positional arguments:
  PATH_TO_DATASET       The relative path to the dataset

optional arguments:
  -h, --help            show this help message and exit
```

-- Flag definitions ---

----

### Part 2 Example
Running:
```
python3 part2_house_value_regression.py -flags
```
Will:
- do something

----

### Extras
- [iris.dat](iris.dat) was used to test the neural network mini-library.
- [housing.csv](housing.csv) is a subdataset the "1990 California Housing" dataset, used to train the regression neural network.
- [part2_model.pickle](part2_model.pickle]) was created using the regression neural network, trained on the entire dataset with fine-tuned hyperparameters.
