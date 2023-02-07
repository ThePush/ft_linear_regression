import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import ml_toolkit as ml


class LinearRegression:
    '''
    A class that performs a linear regression on a dataset.
    No machine learning libraries are used.

    Parameters:
        dataset_path (str): path to the dataset

    Attributes:
        dataset_path (str): path to the dataset
        learning_rate (float): learning rate for gradient descent
        n_epochs (int): number of epochs for gradient descent
        theta (list): list of theta0 and theta1
        normalized_theta (list): list of normalized theta0 and theta1
        costs (list): list of costs for each epoch
        thetas_history (list): list of thetas for each epoch
        first_col (str): name of the first column
        second_col (str): name of the second column
        df (DataFrame): DataFrame of the dataset
        X (list): list of normalized x values
        Y (list): list of normalized y values

    Methods:
        check_dataset(df): check if the dataset is valid
        find_best_learning_rate(X, Y): find the best learning rate for gradient descent
        gradient_descent(theta0, theta1, X, Y, learning_rate): perform gradient descent
        save_thetas(): save thetas in a file named "thetas.csv"
        plot_data(): plot the data
        print_stats(): print the stats
    '''

    def __init__(self, dataset_path: str):
        self.dataset_path: str = dataset_path
        self.learning_rate: float = 0
        self.n_epochs: int = 0
        self.theta = []
        self.normalized_theta = []
        self.costs = []
        self.thetas_history = []
        self.first_col = ''
        self.second_col = ''
        self.df = pd.DataFrame()
        self.X = []
        self.Y = []
        try:
            filename = self.dataset_path
        except Exception as e:
            sys.exit(e)
        ml.check_datafile(filename)
        self.df = pd.read_csv(filename)
        self.check_dataset(self.df)
        self.first_col = self.df.columns[0]
        self.second_col = self.df.columns[1]

        # Normalize data to be between 0 and 1
        self.X = ml.normalize_array(self.df[self.first_col].values)
        self.Y = ml.normalize_array(self.df[self.second_col].values)
        # Initial values for theta0 and theta1
        self.normalized_theta.append(.0)
        self.normalized_theta.append(.0)
        # Find best learning rate
        self.find_best_learning_rate(self.X, self.Y)
        # Save best stats
        self.normalized_theta[0], self.normalized_theta[1], self.costs, self.thetas_history, self.n_epochs = self.gradient_descent(
            self.normalized_theta[0], self.normalized_theta[1], self.X, self.Y, self.learning_rate)
        # Denormalize theta0 and theta1
        self.theta.append(.0)
        self.theta.append(.0)
        self.theta[0], self.theta[1] = self.denormalize_theta(
            self.normalized_theta[0], self.normalized_theta[1], self.df[self.first_col].values, self.df[self.second_col].values)

    def find_best_learning_rate(self, X: list, Y: list) -> None:
        '''
        Find the best learning rate for gradient descent.

        Parameters:
            X (list): list of x, the inputs
            Y (list): list of y, the actual values
        '''
        best_cost = float('inf')
        # Try all learning rates between 0.0001 and 0.1
        for current_learning_rate in np.arange(0.0001, 0.1, 0.0001):
            _, _, costs, _, _ = self.gradient_descent(
                0, 0, X, Y, current_learning_rate)
            if costs[-1] < best_cost:
                # Store the learning rate that gives the lowest cost
                self.learning_rate = current_learning_rate
                best_cost = costs[-1]

    @staticmethod
    def gradient_descent(theta0: float, theta1: float, X: list, Y: list, learning_rate: float, print_results: bool = False):
        '''
        Perform gradient descent.

        Parameters:
            theta0 (float): b, the y-intercept
            theta1 (float): m, the slope
            X (list): list of x, the inputs
            Y (list): list of y, the actual values
            learning_rate (float): the learning rate
            print_results (bool, optional): whether to print the results or not

        Returns:
            theta0 (float): b, the y-intercept
            theta1 (float): m, the slope
            costs (list): list of costs
            thetas_history (list): list of tuples of theta0 and theta1
            n_epochs (int): number of epochs
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
            costs.append(ml.cost(theta0, theta1, X, Y))
            # Calculate the gradient for theta0 and theta1
            # The gradient is the partial derivative of the sum of squared errors
            gradient0 = sum(2 * ml.error(theta0, theta1, x_i, y_i)
                            for x_i, y_i in zip(X, Y))
            gradient1 = sum(2 * ml.error(theta0, theta1, x_i, y_i) * x_i
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
        print(
            f'Number of epochs: {number_of_epochs}') if print_results else None
        print(f'Cost: {costs[-1]}') if print_results else None
        return theta0, theta1, costs, thetas_history, number_of_epochs

    def plot_data(self) -> None:
        '''
        Plot:
            • Raw data as a scatterplot

            • Standardized data and regression line

            • Cost evolution

            • Visualization of the gradient descent algorithm in 3d with theta0, theta1 and cost function
        '''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot raw data
        axes[0].set_title(self.second_col + ' vs ' + self.first_col)
        axes[0].scatter(self.df[self.first_col].values,
                        self.df[self.second_col].values)
        axes[0].set_xlabel(self.first_col)
        axes[0].set_ylabel(self.second_col)
        # Plot standardized data and regression line
        axes[1].set_title('Standardized data and regression line')
        axes[1].scatter(self.X, self.Y)
        axes[1].set_xlabel(self.first_col)
        axes[1].set_ylabel(self.second_col)
        axes[1].plot(self.X, [ml.predict(self.normalized_theta[0], self.normalized_theta[1], x_i)
                     for x_i in self.X], color='red')
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
                cost_mesh[i, j] = ml.cost(theta[0], theta[1], self.X, self.Y)
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
        '''
        print(
            f'LINEAR REGRESSION for dataset: {self.second_col} vs {self.first_col}')
        print(f'\ttheta0: {self.theta[0]}')
        print(f'\ttheta1: {self.theta[1]}')
        print(
            f'\tMODEL: {self.second_col} = {self.theta[0]} + {self.theta[1]} * {self.first_col}')
        print(f'\nMODEL PERFORMANCE:')
        print(
            f'\tMean squared error: {ml.mean_squared_error(self.normalized_theta[0], self.normalized_theta[1], self.X, self.Y)}')
        print(f'\tCost: {self.costs[-1]}')
        print(f'\tAccuracy(%): {100 - self.costs[-1] * 100}')
        print(f'\tLearning rate: {self.learning_rate}')
        print(f'\tNumber of epochs: {self.n_epochs}')
        print(
            f'\tStandard error of the estimate(0,1): {ml.std_err_of_estimate(self.normalized_theta[0], self.normalized_theta[1], self.X, self.Y)}')
        print(
            f'\tStandard error of the estimate(dependent variable): {ml.std_err_of_estimate(self.theta[0], self.theta[1], self.df[self.first_col], self.df[self.second_col])}')
        print(f'\nDATASET STATISTICS:')
        print(
            f'\tPearson Correlation(-1,1): {ml.correlation(self.X, self.Y)}')  # aka r
        print(
            f'\tCoefficient of determination(0,1): {ml.r_squared(self.X, self.Y)}')  # aka r^2

    def save_thetas(self, filename: str = 'theta.csv') -> None:
        '''
        Write theta0 and theta1 to a csv file.

        Arg(optional):
            filename (str): Name of the file to write to. Default is 'theta.csv'
        '''
        if not filename.endswith('.csv'):
            filename += '.csv'
        with open(filename, 'w') as f:
            f.write(f'{self.theta[0]},{self.theta[1]}')
