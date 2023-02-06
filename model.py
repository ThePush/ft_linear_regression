import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


# Model
def predict(theta0: float, theta1: float, x_i: float) -> float:
    '''
    Linear function of type y = mx + b.
    theta0: b, the y-intercept
    theta1: m, the slope
    x_i: x, the input
    '''
    return theta1 * x_i + theta0


def error(theta0: float, theta1: float, x_i: float, y_i: float) -> float:
    '''
    Difference between predicted and actual value.
    theta0: b, the y-intercept
    theta1: m, the slope
    x_i: x, the input
    y_i: y, the actual value
    '''
    return predict(theta0, theta1, x_i) - y_i


def sum_of_squared_errors(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Sum of squared errors between predicted and actual values.
    theta0: b, the y-intercept
    theta1: m, the slope
    X: list of x, the inputs
    Y: list of y, the actual values
    '''
    return sum(error(theta0, theta1, x_i, y_i) ** 2
               for x_i, y_i in zip(X, Y))


def cost(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Average squared error between predicted and actual values.
    theta0: b, the y-intercept
    theta1: m, the slope
    X: list of x, the inputs
    Y: list of y, the actual values
    '''
    return sum_of_squared_errors(theta0, theta1, X, Y) / (2 * len(X))


def find_best_learning_rate(X: list, Y: list) -> float:
    '''
    Find the best learning rate for gradient descent.
    X: list of x, the inputs
    Y: list of y, the actual values
    '''
    best_learning_rate: float = 0
    best_cost = float('inf')
    # Try all learning rates between 0.0001 and 0.1
    for current_learning_rate in np.arange(0.0001, 0.1, 0.0001):
        _, _, costs, _, _ = gradient_descent(
            0, 0, X, Y, current_learning_rate, False)
        if costs[-1] < best_cost:
            # Store the learning rate that gives the lowest cost
            best_learning_rate = current_learning_rate
            best_cost = costs[-1]
    return best_learning_rate


def gradient_descent(theta0: float, theta1: float, X: list, Y: list, learning_rate: float, print_results: bool = True) -> tuple:
    '''
    Update theta0 and theta1 using gradient descent algorithm.
    theta0: b, the y-intercept
    theta1: m, the slope
    X: list of x, the inputs
    Y: list of y, the actual values
    learning_rate: the learning rate
    print_results: bool to know if the results should be printed
    '''
    # Number of maximum iterations to perform gradient descent
    num_epochs = 100_000
    # Variable for plotting and measures
    number_of_epochs: int = 0
    thetas_history = []
    costs = []

    # Core of the gradient descent algorithm
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
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1
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
    print(f'Number of epochs: {number_of_epochs}') if print_results else None
    print(f'Cost: {costs[-1]}') if print_results else None
    return theta0, theta1, costs, thetas_history, number_of_epochs


# Plotting
def plot_data(
        X: np.array, Y: np.array, theta0: float, theta1: float, x: np.array, y: np.array,
        costs: list, thetas_history: list, first_col: str, second_col: str) -> None:
    '''Plot:
    1/ Raw data as a scatterplot
    2/ Standardized data and regression line
    3/ Cost evolution
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot raw data
    axes[0].set_title(second_col + ' vs ' + first_col)
    axes[0].scatter(x, y)
    axes[0].set_xlabel(first_col)
    axes[0].set_ylabel(second_col)
    # Plot standardized data and regression line
    axes[1].set_title('Standardized data and regression line')
    axes[1].scatter(X, Y)
    axes[1].set_xlabel(first_col)
    axes[1].set_ylabel(second_col)
    axes[1].plot(X, [predict(theta0, theta1, x_i) for x_i in X], color='red')
    # Plot cost
    axes[2].set_title('Cost evolution')
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
    theta0_min, theta0_max = min(
        [x[0] for x in thetas_history])-.2, max([x[0] for x in thetas_history])+.2
    theta1_min, theta1_max = min(
        [x[1] for x in thetas_history])-.2, max([x[1] for x in thetas_history])+.2
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
    ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh,
                    cmap='coolwarm', alpha=0.5)
    ax.scatter([x[0] for x in thetas_history], [x[1]
               for x in thetas_history], costs, s=1, color='purple', alpha=1)
    ax.plot([x[0] for x in thetas_history], [x[1]
            for x in thetas_history], costs, color='purple', alpha=1)
    plt.show()


# Utils
def correlation(x: pd.DataFrame, y: pd.DataFrame) -> float:
    '''Calculate the Pearson (linear) correlation between two variables.'''
    n = len(x)
    numerator = n * sum(x * y) - sum(x) * sum(y)
    denominator = ((n * sum(x**2) - sum(x)**2)**.5) * \
        ((n * sum(y**2) - sum(y)**2)**.5)
    return numerator / denominator


def std_err_of_estimate(theta0: float, theta1: float, X: list, Y: list) -> float:
    '''
    Standard error of estimate.
    theta0: b, the y-intercept
    theta1: m, the slope
    X: list of x, the inputs
    Y: list of y, the actual values
    '''
    return (sum_of_squared_errors(theta0, theta1, X, Y) / (len(X) - 2))**.5


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


def check_datafile(filename: str):
    '''Check if <file.csv> exists and is not empty'''
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError
        if os.stat(filename).st_size == 0:
            raise ValueError
    except FileNotFoundError:
        sys.exit(
            'Please provide ' + filename + ' in the same directory as this script')
    except ValueError:
        sys.exit('File ' + filename + ' is empty')


def check_dataset(dataset: pd.DataFrame):
    '''Check if dataset is valid'''
    try:
        if len(dataset.columns) != 2:
            raise Exception
        if dataset.isnull().values.any():
            raise Exception
        for col in dataset.columns:
            if not np.issubdtype(dataset[col].dtype, np.number):
                raise Exception
    except Exception:
        sys.exit('Invalid dataset')


def main():
    '''
    A program that performs a linear regression on a dataset.
    No external libraries are used.
    Usage: python3 model.py <file.csv>
    '''
    try:
        filename = sys.argv[1]
    except IndexError:
        sys.exit('Please provide a dataset <file.csv> as an argument')
    check_datafile(filename)
    df = pd.read_csv(filename)
    df = df.dropna()
    check_dataset(df)
    first_col = df.columns[0]
    second_col = df.columns[1]

    # Normalize data to be between 0 and 1
    X = normalize_array(df[first_col].values)
    Y = normalize_array(df[second_col].values)
    # Initial values for theta0 and theta1
    theta = []
    theta.append(.0)
    theta.append(.0)
    # Find best learning rate
    L = find_best_learning_rate(X, Y)
    # Perform gradient descent
    theta[0], theta[1], costs, thetas_history, n_epochs = gradient_descent(
        theta[0], theta[1], X, Y, L, False)
    # Plot data
    plot_data(X, Y, theta[0], theta[1],
              df[first_col].values, df[second_col].values, costs, thetas_history, first_col, second_col)
    # Denormalize theta0 and theta1
    theta[0], theta[1] = denormalize_theta(
        theta[0], theta[1], df[first_col].values, df[second_col].values)

    # Print stats
    print(f'LINEAR REGRESSION for dataset: {second_col} vs {first_col}')
    print(f'\ttheta0: {theta[0]}')
    print(f'\ttheta1: {theta[1]}')
    print(f'\tMODEL: {second_col} = {theta[0]} + {theta[1]} * {first_col}')
    print(f'\nPERFORMANCE:')
    print(f'\tBest cost: {costs[-1]}')
    print(f'\tBest learning rate: {L}')
    print(f'\tNumber of epochs: {n_epochs}')
    print(f'\tAccuracy(%): {100 - costs[-1] * 100}')
    print(
        f'\tStandard error of the estimate(€): {std_err_of_estimate(theta[0], theta[1], df[first_col].values, df[second_col].values)}')
    print(f'\nDATASET STATISTICS:')
    print(
        f'\tPearson Correlation(-1,1): {correlation(df[first_col].values, df[second_col].values)}')  # aka r
    print(
        f'\tCoefficient of determination(0,1): {correlation(df[first_col].values, df[second_col].values)**2}')  # aka squared correlation or r^2

    # Write theta0 and theta1 to csv file
    with open('theta.csv', 'w') as f:
        f.write(f'{theta[0]},{theta[1]}')


if __name__ == '__main__':
    main()
