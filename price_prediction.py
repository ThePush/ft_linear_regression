import os
import sys


def main():
    x = input('Enter a kilometer value: ')
    if not x.isdigit() or int(x) < 0:
        print('Invalid input, please enter a positive integer')
        sys.exit(1)

    # Initialize theta0 and theta1 that will be used to predict the price
    theta0, theta1 = .0, .0
    # Load the theta values from the file
    if os.path.exists('theta.csv'):
        with open('theta.csv', 'r') as f:
            theta0, theta1 = f.read().split(',')
            theta0 = float(theta0)
            theta1 = float(theta1)

    # Linear function type: y = ax + b
    price = theta0 + theta1 * float(x)
    print(f'Predicted price: {price}')


if __name__ == '__main__':
    main()
