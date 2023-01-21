import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


# Model
def predict(theta0: float, theta1: float, x_i: float) -> float:
    '''Linear function of type y = ax + b'''
    return theta1 * x_i + theta0


def error(theta0: float, theta1: float, x_i: float, y_i: float) -> float:
    '''Difference between predicted and actual value'''
    return predict(theta0, theta1, x_i) - y_i


def sum_of_squared_errors(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''Sum of squared errors between predicted and actual values'''
    return sum(error(theta0, theta1, x_i, y_i) ** 2
               for x_i, y_i in zip(X, Y))


def cost(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''Average squared error between predicted and actual values'''
    return sum_of_squared_errors(theta0, theta1, X, Y) / (2 * len(X))


def gradient_descent(theta0: float, theta1: float, X: list, Y: list) -> tuple:
    '''Update theta0 and theta1 using gradient descent algorithm'''
    # Number of maximum iterations to perform gradient descent
    num_epochs = 100_000
    # Learning rate
    L = .035
    # Variable for plotting and measures
    number_of_epochs = 0
    thetas_history = []
    costs = []
    mse = []
    for _ in range(num_epochs):
        # Store values for plotting
        thetas_history.append((theta0, theta1))
        costs.append(cost(theta0, theta1, X, Y))

        # Calculate the gradient for theta0 and theta1
        # The gradient is the partial derivative of the sum of squared errors
        gradient0 = sum(2 * error(theta0, theta1, x_i, y_i)
                        for x_i, y_i in zip(X, Y))
        gradient1 = sum(2 * error(theta0, theta1, x_i, y_i) * x_i
                        for x_i, y_i in zip(X, Y))
        # Update theta0 and theta1
        theta0 -= L * gradient0
        theta1 -= L * gradient1
        number_of_epochs += 1
        # Stop if:
        # theta0 and theta1 have converged or
        # if the previous cost is less than the current cost or
        # if the previous cost is different than the current cost by less than 0.000001
        if theta0 == thetas_history[-1][0] and theta1 == thetas_history[-1][1] or \
            ((len(costs) > 1 and
                (costs[-2] < costs[-1] or
                 abs(costs[-2] - costs[-1]) < 0.000001))):
            break
    print(f'Number of epochs: {number_of_epochs}')
    print(f'Final cost: {costs[-1]}')
    return theta0, theta1, costs, mse, thetas_history


def normalize_array(X: np.array) -> np.array:
    '''Normalize data to be between 0 and 1'''
    return (X - X.min()) / (X.max() - X.min())


def denormalize_theta(theta0: float, theta1: float, X: np.array, Y: np.array) -> tuple:
    '''Denormalize theta0 and theta1'''
    x_min = X.min()
    x_max = X.max()
    y_min = Y.min()
    y_max = Y.max()
    return theta0 * (y_max - y_min) + y_min, theta1 * (y_max - y_min) / (x_max - x_min)


def plot_data(
        X: np.array, Y: np.array, theta0: float, theta1: float, x: np.array, y: np.array,
        costs: list, thetas_history: list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot raw data
    axes[0].set_title('Car price by mileage')
    axes[0].scatter(x, y)
    axes[0].set_xlabel('km')
    axes[0].set_ylabel('price')
    # Plot standardized data and regression line
    axes[1].set_title('Standardized data and regression line')
    axes[1].scatter(X, Y)
    axes[1].set_xlabel('km')
    axes[1].set_ylabel('price')
    axes[1].plot(X, [predict(theta0, theta1, x_i) for x_i in X], color='red')
    # Plot cost
    axes[2].set_title('Cost')
    axes[2].plot(costs)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Cost')
    # Plot gradient descent with thetas and cost function as surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Gradient descent visualization')
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('Cost')
    # Find the range of theta0 and theta1 values that the theta descent path covers
    theta0_min, theta0_max = min([x[0] for x in thetas_history])-.2, max([x[0] for x in thetas_history])+.2
    theta1_min, theta1_max = min([x[1] for x in thetas_history])-.2, max([x[1] for x in thetas_history])+.2
    # Create a smaller meshgrid for the surface plot
    theta0_vals = np.linspace(theta0_min, theta0_max, 100)
    theta1_vals = np.linspace(theta1_min, theta1_max, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    # Compute the cost function for each pair of theta0 and theta1 values
    cost_mesh = np.zeros_like(theta0_mesh)
    for i in range(theta0_mesh.shape[0]):
        for j in range(theta0_mesh.shape[1]):
            theta = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
            cost_mesh[i, j] = cost(theta[0], theta[1], X, Y)
    # Plot the cost function surface
    ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh, cmap='coolwarm', alpha=0.5)
    ax.scatter([x[0] for x in thetas_history], [x[1] for x in thetas_history], costs, s=1, color='purple', alpha=1)
    ax.plot([x[0] for x in thetas_history], [x[1] for x in thetas_history], costs, color='purple', alpha=1)

    plt.show()


def main():
    '''Main function'''
    # Check if data.csv exists and is not empty
    assert os.path.exists('data.csv'), sys.exit(
        'Please download data.csv from intra and place it in the same directory as this script')
    assert os.stat('data.csv').st_size != 0, sys.exit('File is empty')

    # Read dataset from csv file
    data = pd.read_csv('data.csv')
    # Check if dataset is valid
    try:
        data['km'].values
        data['price'].values
        for x in data['km'].values:
            float(x)
        for y in data['price'].values:
            float(y)
    except Exception:
        sys.exit('Invalid dataset')

    # Normalize data to be between 0 and 1
    X = normalize_array(data['km'].values)
    Y = normalize_array(data['price'].values)
    # Initial values for theta0 and theta1
    theta = []
    theta.append(.0)
    theta.append(.0)
    # Perform gradient descent
    theta[0], theta[1], costs, mse, thetas_history = gradient_descent(
        theta[0], theta[1], X, Y)
    # Plot data
    plot_data(X, Y, theta[0], theta[1],
              data['km'].values, data['price'].values, costs, thetas_history)

    # Denormalize theta0 and theta1
    theta[0], theta[1] = denormalize_theta(
        theta[0], theta[1], data['km'].values, data['price'].values)
    # Print final values for theta0 and theta1
    print(f'theta0: {theta[0]}')
    print(f'theta1: {theta[1]}')
    # Write theta0 and theta1 to csv file
    with open('theta.csv', 'w') as f:
        f.write(f'{theta[0]},{theta[1]}')


if __name__ == '__main__':
    main()
