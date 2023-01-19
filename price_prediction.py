import sys


def main():
    user_input = input('Enter a kilometer value: ')
    if not user_input.isdigit() or int(user_input) < 0:
        print('Invalid input, please enter a positive integer')
        sys.exit(1)

    # Load the theta values from the file
    with open('theta.csv', 'r') as f:
        theta0, theta1 = f.read().split(',')
        theta0 = float(theta0)
        theta1 = float(theta1)

    price = theta0 + theta1 * float(user_input)
    print(f'Predicted price: {price}')


if __name__ == '__main__':
    main()
