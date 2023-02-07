import sys
from LinearRegression import LinearRegression


def print_help() -> None:
    '''Print the help message of the program.'''
    print('USAGE:')
    print('\t$> python3 linreg.py <dataset_path> [-p | --plot] [-s | --stats] [-h | --help]')
    print('DESCRIPTION:')
    print('\tThis program performs linear regression on a dataset, user must provide a dataset path.')
    print('\tThe dataset must be a CSV file with two columns, the first column is the input (x) and the second column is the output (y).')
    print('Example of dataset:')
    print('\tx,y')
    print('\t1,2')
    print('\t2,4')
    print('\t3,6')
    print('\tThe program will save the thetas (b and m) in a file named "thetas.csv"')
    print('OPTIONS:')
    print('\t-p, --plot\t\tPlot the data')
    print('\t-s, --stats\t\tPrint the stats')
    print('\t-h, --help\t\tPrint this help')
    print('EXAMPLE:')
    print('\t$> python3 linreg.py data.csv -ps')


def parse_user_input() -> dict:
    '''
    Parse user input and return a dictionary of options.

    Parameters:
        None

    Returns:
        A dictionary of options.
    '''
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 linreg.py <dataset_path> [-p, plot] [-s, stats] [-h, help]')
    if sys.argv[1].startswith('-'):
        print_help()
        sys.exit()
    options = {'p': False, 's': False, 'h': False}
    for arg in sys.argv[2:]:
        if arg.startswith('-'):
            stripped_arg = arg.lstrip('-')
            concatenated_options = ''.join([c for c in stripped_arg if c in options])
            if len(concatenated_options) != len(stripped_arg):
                invalid_options = [c for c in stripped_arg if c not in options]
                print(f'Invalid option(s): {", ".join(invalid_options)}')
            for option in concatenated_options:
                options[option] = True
        else:
            print(f'Invalid option: {arg}')
    return options


def main():
    '''
    linreg.py is a program that performs linear regression on a dataset.
    Usage: python3 linreg.py <dataset_path> [-p, plot] [-s, stats] [-h, help]
    '''
    options = parse_user_input()
    try:
        lr = LinearRegression(sys.argv[1])
        lr.save_thetas()
        if options['p']:
            lr.plot_data()
        if options['s']:
            lr.print_stats()
        if options['h']:
            print_help()
    except Exception as e:
        sys.exit(f'Error: {e}')

if __name__ == '__main__':
    main()
