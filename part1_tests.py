import numpy as np
import part1_nn_lib as nn


def linear():
    print("--------------------")
    print("Testing Linear Layer")
    print("--------------------")
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = nn.Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    print("\n>> Testing Constructor\n")
    layer = nn.LinearLayer(4, 4)
    print(layer._W)
    print(layer._b)
    print(layer._cache_current)
    assert layer._grad_W_current.shape == layer._W.shape
    assert layer._grad_b_current.shape == layer._b.shape

    print("\n>> Testing Forward Propagation\n")
    d_in = x_train_pre[:4, :]
    out = layer(d_in)
    print(out)
    assert np.isclose(layer._cache_current.transpose(), d_in).all()

    print("\n>> Testing Backwards Propagation")
    grad_z = np.array(
        [
            [1, -3, 4, 1.2],
            [0.6, -2.7, 3.3, -0.3],
            [1.2, -5.2, 1.9, 2.4],
            [2.8, -2.9, 5.1, -1.2],
        ]
    )
    print("\n- Previous Gradient Values:")
    print(layer._grad_W_current)
    print(layer._grad_b_current)
    grad_x = layer.backward(grad_z)
    assert grad_x.shape == (4, 4)
    print("\n- New Gradient Values:")
    print(layer._grad_W_current)
    print(layer._grad_b_current)

    print("\n>> Testing update_params Function")
    print("\n- Previous param:")
    print(layer._W)
    print(layer._b)
    layer.update_params(0.001)
    print("\n- New param:")
    print(layer._W)
    print(layer._b)


def activation():
    print("-------------------------")
    print("Testing Activation Layer")
    print("-------------------------")
    print("\n>> Testing Sigmoid Activation")
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = nn.Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    print("\n> Testing Constructor")
    layer = nn.SigmoidLayer()

    print("\n> Testing Forward")
    d_in = x_train_pre[:4, :]
    out = layer(d_in)
    print(out)
    assert np.isclose(layer._cache_current, d_in).all()
    assert np.isclose(out, np.reciprocal(np.exp(-d_in) + 1)).all()

    print("\n> Testing Backward")
    grad_z = np.array(
        [
            [1, -3, 4, 1.2],
            [0.6, -2.7, 3.3, -0.3],
            [1.2, -5.2, 1.9, 2.4],
            [2.8, -2.9, 5.1, -1.2],
        ]
    )
    grad_x = layer.backward(grad_z)
    assert grad_x.shape == grad_z.shape
    print(grad_x)
    sig = np.reciprocal(np.exp(-layer._cache_current) + 1)
    assert np.isclose(grad_x, grad_z * sig * (1 - sig)).all()

    print("\n\n>> Testing Relu Activation")

    print("\n> Testing Constructor")
    layer = nn.ReluLayer()

    print("\n> Testing Forward")
    d_in = x_train_pre[:4, :]
    out = layer(d_in)
    print(out)
    assert out.shape == d_in.shape
    assert np.isclose(layer._cache_current, d_in).all()
    test_out = d_in
    test_out[test_out < 0] = 0

    assert np.isclose(out, test_out).all()

    print("\n> Testing Backward")
    grad_z = np.array(
        [
            [1, -3, 4, 1.2],
            [0.6, -2.7, 3.3, -0.3],
            [1.2, -5.2, 1.9, 2.4],
            [2.8, -2.9, 5.1, -1.2],
        ]
    )
    grad_x = layer.backward(grad_z)
    assert grad_x.shape == grad_z.shape
    print(grad_x)
    assert np.isclose(grad_x, grad_z * (layer._cache_current > 0).astype(int)).all()


def multilayer():
    print("-------------------------")
    print("Testing Multi Layer Network")
    print("-------------------------")
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = nn.Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    print("\n>> Testing Constructor\n")
    network = nn.MultiLayerNetwork(
        input_dim=4, neurons=[16, 3], activations=["relu", "sigmoid"]
    )

    EPOCHS = 100
    learning_rate = 0.01

    for _ in range(EPOCHS):
        y_pred = network.forward(x_train)
        grad_z = y_pred - y_train

        network.backward(grad_z)
        network.update_params(learning_rate)


if __name__ == "__main__":
    print("1 - Linear Layer Test")
    print("2 - Activation Layer Test")
    print("3 - Multi-Layer Network Test")
    option = input()
    if option == "1":
        linear()
    elif option == "2":
        activation()
    elif option == "3":
        multilayer()
    else:
        print("invalid option")
