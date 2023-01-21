import os
import sys


def main():
    '''Main function'''
    x = input('Enter a kilometer value: ')
    assert x, sys.exit('Please enter a kilometer value')
    assert x.isnumeric() and float(x) >= 0, sys.exit(
        'Please enter a valid kilometer value')

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

    # Linear function type: y = ax + b
    price = theta0 + theta1 * float(x)
    print(f'Predicted price: {price}')


if __name__ == '__main__':
    main()
