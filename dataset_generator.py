import csv
import random


def generate_dataset():
    '''Generate dataset'''
    with open('houses_prices.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m2', 'price'])

        for i in range(50):
            m2 = random.randint(50, 200)
            price = m2 * random.uniform(1000, 5000)
            writer.writerow([m2, price])


def main():
    '''Main function'''
    generate_dataset()


if __name__ == '__main__':
    main()
