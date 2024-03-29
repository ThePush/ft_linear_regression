import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import srcs.ml_toolkit as ml
from tqdm import tqdm


class LinearRegression:
    '''
    A class that performs a linear regression on a dataset.
    No machine learning or math libraries are used but mines.

    Parameters:
        dataset_path (str): path to the dataset

    Attributes:
        filename (str): path to the dataset
        learning_rate (float): learning rate for gradient descent
        n_epochs (int): number of epochs for gradient descent
        theta (list): list of theta0 and theta1
        normalized_theta (list): list of normalized theta0 and theta1
        thetas_history (list): list of thetas for each epoch
        first_col (str): name of the first column
        second_col (str): name of the second column
        df (DataFrame): DataFrame of the dataset
        X (np.ndarray): list of x, the inputs
        Y (np.ndarray): list of y, the actual values

    Methods:
        check_dataset(df): check if the dataset is valid
        find_best_learning_rate(X, Y): find the best learning rate for gradient descent
        gradient_descent(theta0, theta1, X, Y, learning_rate): perform gradient descent
        fit(): fit the model
        save_thetas(): save thetas in a file named "thetas.csv"
        plot_data(): plot the data
        print_stats(): print the stats
    '''

    def __init__(self, filename: str, n_epochs: int = 1000):
        try:
            assert isinstance(
                filename, str), 'filename must be a string'
            assert isinstance(n_epochs, int), 'n_epochs must be an integer'
            assert n_epochs > 0, 'n_epochs must be greater than 0'
            ml.check_datafile(filename)
            self.df = pd.read_csv(filename)
            self.check_dataset(self.df)
        except Exception as e:
            print(e)
            sys.exit(1)

        self.learning_rate: float = 0
        self.n_epochs: int = n_epochs
        self.first_col = self.df.columns[0]
        self.second_col = self.df.columns[1]

        self.X = self.df[self.first_col].to_numpy()
        self.Y = self.df[self.second_col].to_numpy()
        self.X_norm = ml.normalize_array(self.X)
        self.Y_norm = ml.normalize_array(self.Y)
        self.theta = []
        self.theta.append(.0)
        self.theta.append(.0)
        self.normalized_theta = []
        self.normalized_theta.append(.0)
        self.normalized_theta.append(.0)

    def find_best_learning_rate(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Find the best learning rate for gradient descent.

        Parameters:
            X (np.ndarray): list of x, the inputs
            Y (np.ndarray): list of y, the actual values

        Returns:
            None
        '''
        print('Finding the best learning rate...')
        best_cost = float('inf')
        # Try all learning rates between 0.0001 and 0.1
        for current_learning_rate in tqdm(np.arange(0.0001, 0.1, 0.0001), total=999):
            _, _, costs, _, _ = self.gradient_descent(
                0, 0, X, Y, current_learning_rate)
            if costs[-1] < best_cost:
                # Store the learning rate that gives the lowest cost
                self.learning_rate = current_learning_rate
                best_cost = costs[-1]
        print(f'Best learning rate: {self.learning_rate}')

    def gradient_descent(self, theta0: float, theta1: float, X: np.ndarray, Y: np.ndarray, learning_rate: float, print_results: bool = False):
        '''
        Perform gradient descent.

        Parameters:
            theta0 (float): b, the y-intercept
            theta1 (float): m, the slope
            X (np.ndarray): list of x, the inputs
            Y (np.ndarray): list of y, the actual values
            learning_rate (float): the learning rate
            print_results (bool, optional): whether to print the results or not

        Returns:
            theta0 (float): b, the y-intercept
            theta1 (float): m, the slope
            costs (list): list of costs
            thetas_history (list): list of tuples of theta0 and theta1
            n_epochs (int): number of epochs
        '''
        # Variable for plotting and measures
        thetas_history = []
        costs = []
        # Create a matrix to store thetas
        thetas = np.array([theta0, theta1], dtype=np.float64)
        # Core of the gradient descent algorithm
        for i in range(self.n_epochs):
            # Store values for plotting
            thetas_history.append((thetas[0], thetas[1]))
            costs.append(ml.cost(thetas[0], thetas[1], X, Y))
            # Calculate the gradient for theta0 and theta1
            # The gradient is the partial derivative of the sum of squared errors
            errors = ml.error(thetas[0], thetas[1], X, Y)
            gradient = np.array(
                [2 * np.sum(errors), 2 * np.dot(errors, X)], dtype=np.float64)
            # Update thetas using vectorization
            thetas -= learning_rate * gradient
            print(
                f'Epoch: {i + 1}\nCost: {costs[-1]}') if print_results else None
            # Stop if:
            # thetas have converged or
            # if the previous cost is less than the current cost or
            # if the previous cost is different than the current cost by less than 0.000001
            if np.allclose(thetas, thetas_history[-1]) or \
                    ((len(costs) > 1 and (costs[-2] < costs[-1] or
                                          abs(costs[-2] - costs[-1]) < 0.000001))):
                break
        return thetas[0], thetas[1], costs, thetas_history, i+1

    def fit(self):
        self.find_best_learning_rate(self.X_norm, self.Y_norm)
        self.normalized_theta[0], self.normalized_theta[1], self.costs, self.thetas_history, self.n_epochs = self.gradient_descent(
            self.normalized_theta[0], self.normalized_theta[1], self.X_norm, self.Y_norm, self.learning_rate, True)
        self.theta[0], self.theta[1] = self.denormalize_theta(
            self.normalized_theta[0], self.normalized_theta[1], self.X, self.Y)

    def plot_data(self) -> None:
        '''
        Plot:
            • Raw data as a scatterplot

            • Standardized data and regression line

            • Cost evolution

            • Visualization of the gradient descent algorithm in 3d with theta0, theta1 and cost function

        Parameters:
            None

        Returns:
            None
        '''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot raw data
        axes[0].set_title(self.second_col + ' vs ' + self.first_col)
        axes[0].scatter(self.X,
                        self.Y)
        axes[0].set_xlabel(self.first_col)
        axes[0].set_ylabel(self.second_col)
        # Plot standardized data and regression line
        axes[1].set_title('Standardized data and regression line')
        axes[1].scatter(self.X_norm, self.Y_norm)
        axes[1].set_xlabel(self.first_col)
        axes[1].set_ylabel(self.second_col)
        axes[1].plot(self.X_norm, [ml.predict(self.normalized_theta[0], self.normalized_theta[1], x_i)
                     for x_i in self.X_norm], color='red')
        # Plot cost
        axes[2].set_title('Cost evolution')
        axes[2].plot(self.costs)
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
            [x[0] for x in self.thetas_history])-.2, max([x[0] for x in self.thetas_history])+.2
        theta1_min, theta1_max = min(
            [x[1] for x in self.thetas_history])-.2, max([x[1] for x in self.thetas_history])+.2
        # Create a smaller meshgrid for the surface plot
        theta0_vals = np.linspace(theta0_min, theta0_max, 100)
        theta1_vals = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
        # Compute the cost function for each pair of theta0 and theta1 values
        cost_mesh = np.zeros_like(theta0_mesh)
        for i in range(theta0_mesh.shape[0]):
            for j in range(theta0_mesh.shape[1]):
                theta = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
                cost_mesh[i, j] = ml.cost(theta[0], theta[1], self.X_norm, self.Y_norm)
        # Plot the cost function surface
        ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh,
                        cmap='coolwarm', alpha=0.5)
        ax.scatter([x[0] for x in self.thetas_history], [x[1]
                                                         for x in self.thetas_history], self.costs, s=1, color='purple', alpha=1)
        ax.plot([x[0] for x in self.thetas_history], [x[1]
                for x in self.thetas_history], self.costs, color='purple', alpha=1)
        plt.show()

    @staticmethod
    def denormalize_theta(theta0: float, theta1: float, X: np.array, Y: np.array) -> tuple:
        '''
        Denormalize theta0 and theta1

        Parameters:
            theta0 (float): Normalized theta0
            theta1 (float): Normalized theta1
            X (np.array): X values
            Y (np.array): Y values

        Returns:
            theta0, theta1 (tuple): Denormalized theta0 and theta1
        '''
        x_min = X.min()
        x_max = X.max()
        y_min = Y.min()
        y_max = Y.max()
        return theta0 * (y_max - y_min) + y_min, theta1 * (y_max - y_min) / (x_max - x_min)

    @staticmethod
    def check_dataset(df: pd.DataFrame) -> None:
        '''Check if dataset is two columns of numbers with no null values.'''
        if len(df.columns) != 2:
            raise Exception('Dataset must have two columns')
        if df.isnull().values.any():
            raise Exception('Dataset must not have null values')
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise Exception(f'Column {col} is not numeric')

    def print_stats(self) -> None:
        '''
        Print stats about the linear regression and dataset.

        Parameters:
            None

        Returns:
            None
        '''
        print(
            f'LINEAR REGRESSION for dataset: {self.second_col} vs {self.first_col}')
        print(f'\ttheta0: {self.theta[0]}')
        print(f'\ttheta1: {self.theta[1]}')
        print(
            f'\tMODEL: {self.second_col} = {self.theta[0]} + {self.theta[1]} * {self.first_col}')
        print(f'\nMODEL PERFORMANCE:')
        print(
            f'\tMean squared error: {ml.mean_squared_error(self.normalized_theta[0], self.normalized_theta[1], self.X_norm, self.Y_norm)}')
        print(f'\tCost: {self.costs[-1]}')
        print(f'\tAccuracy(%): {100 - self.costs[-1] * 100}')
        print(f'\tLearning rate: {self.learning_rate}')
        print(f'\tNumber of epochs: {self.n_epochs}')
        print(
            f'\tStandard error of the estimate(0,1): {ml.std_err_of_estimate(self.normalized_theta[0], self.normalized_theta[1], self.X_norm, self.Y_norm)}')
        print(
            f'\tStandard error of the estimate(dependent variable): {ml.std_err_of_estimate(self.theta[0], self.theta[1], self.X, self.Y)}')
        print(f'\nDATASET STATISTICS:')
        print(
            f'\tPearson Correlation(-1,1): {ml.correlation(self.X_norm, self.Y_norm)}')  # aka r
        print(
            f'\tCoefficient of determination(0,1): {ml.r_squared(self.X_norm, self.Y_norm)}')  # aka r^2

    def save_thetas(self, filename: str = 'theta.csv') -> None:
        '''
        Write theta0 and theta1 to a csv file.

        Parameters (optional):
            filename (str): Name of the file to write to. Default is 'theta.csv'

        Returns:
            None
        '''
        if not filename.endswith('.csv'):
            filename += '.csv'
        with open(filename, 'w') as f:
            f.write(f'{self.theta[0]},{self.theta[1]}')
