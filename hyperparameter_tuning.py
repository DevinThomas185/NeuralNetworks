from part2_house_value_regression import RegressorHyperParameterSearch
import sys
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Hyperparameter Tuning for Neural Networks",
        description="Hyperparameter Tuning",
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
        "-es",
        "--early_stopping",
        action="store_true",
        help="Use this flag to enable early stopping",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="The number of epochs to run the training for",
        default=100,
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        nargs="+",
        help="The rates at which learning is completed",
        default=None,
        type=float,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        nargs="+",
        help="The sizes for each batch of mini-batched learning",
        default=None,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--dropout",
        nargs="+",
        help="Provide a list of fractions of neurons to drop in training",
        default=None,
        type=float,
    )

    parser.add_argument(
        "-n",
        "--neurons",
        nargs="+",
        help="Provide the hidden layers neurons choices. Please separate choices with a : (e.g 5 5 : 10 10)",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-x",
        "--seed",
        help="Provide a seed for shuffling the dataset",
        default=None,
        type=int,
    )

    parser.add_argument(
        "-k",
        "--folds",
        help="Provide the number of folds for cross validation",
        default=5,
        type=int,
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the resulting model to pickle file",
    )

    args = parser.parse_args(sys.argv[1:])

    params = {}

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size

    if args.dropout is not None:
        params["dropout"] = args.dropout

    if args.early_stopping:
        params["early_stop"] = [True, False]

    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate

    if args.neurons is not None:
        neurons = []
        current = []
        for n in args.neurons:
            if n != ":":
                current.append(int(n))
            else:
                neurons.append(current)
                current = []
        neurons.append(current)

        params["neurons"] = neurons

    RegressorHyperParameterSearch(
        seed=args.seed,
        epochs=args.epochs,
        folds=args.folds,
        path_to_dataset=args.PATH_TO_DATASET,
        output_label=args.OUTPUT_LABEL,
        params=params,
        save=args.save,
    )
