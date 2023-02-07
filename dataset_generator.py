import csv
import random
import sys
import numpy as np
import os


def generate_dataset(filename: str, n_rows: int, theta0: float, theta1: float) -> None:
    '''
    Generate dataset of 2 column with random uniform data.

    Args:
    filename (str): name of the file to generate
    n_rows (int): number of rows to generate
    theta0 (float): intercept of the linear regression
    theta1 (float): slope of the linear regression
    '''
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['independent_value', 'dependent_value'])

        for i in range(n_rows):
            independent_value = random.randint(0, 250000)
            dependent_value = theta0 + theta1 * independent_value
            dependent_value += random.uniform(-1500, 1500)
            writer.writerow([independent_value, int(dependent_value)])


def parse_user_input(argv: list):
    '''
    Parse user input and return the filename and number of rows to generate.

    Args:
    argv (list): list of arguments passed to the program
    '''
    try:
        filename = argv[1]
        n_rows = int(argv[2])
        if not filename or not n_rows or n_rows < 1:
            raise ValueError
        if not filename.endswith('.csv'):
            filename += '.csv'
    except IndexError:
        sys.exit('Please provide a dataset <filename> as an argument')
    except ValueError:
        sys.exit('Usage: python3 dataset_generator.py <filename> <number_of_rows>')
    return filename, n_rows


def main():
    '''
    Generate dataset of 2 column with random uniform data.
    Usage: python3 dataset_generator.py <filename> <number_of_rows>
    '''
    # Initialize theta0 and theta1 that will be used to generate the dataset
    theta0, theta1 = .0, .0
    # Load the theta values from the file
    if os.path.exists('theta.csv'):
        with open('theta.csv', 'r') as f:
            try:
                theta0, theta1 = f.read().split(',')
                theta0 = float(theta0)
                theta1 = float(theta1)
            except ValueError:
                sys.exit('theta.csv does not contain valid values')
    filename, n_rows = parse_user_input(sys.argv)
    generate_dataset(filename, n_rows, theta0, theta1)


if __name__ == '__main__':
    main()
