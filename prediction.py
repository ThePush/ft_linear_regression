import os
import sys


def main():
    '''
    User enters a value for the independent variable (x) and the program predicts
    the value of the dependent variable (y), given the values of theta0 and theta1
    computed by the gradient descent algorithm in the model.py file.
    Usage: python3 price_prediction.py
    '''
    x = input('Enter an independent value: ')
    try:
        if not x or float(x) < 0:
            raise ValueError('Please enter a positive number')
    except ValueError as e:
        sys.exit(e)

    # Initialize theta0 and theta1 that will be used to predict the price
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

    # Linear function type: y = mx + b
    price = theta0 + theta1 * float(x)
    print(f'Predicted dependent value: {price}')


if __name__ == '__main__':
    main()
