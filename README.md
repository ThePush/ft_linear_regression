## Linear Regression: Predicting dependent variables from independent variables


![image](https://user-images.githubusercontent.com/91064070/213888620-6b488e8f-0164-4655-a442-af6fe28b1bb3.png)

![image](https://user-images.githubusercontent.com/91064070/213887705-0a1ac769-e042-4d45-a0c6-717af65efeea.png)


## Usage:

![image](https://user-images.githubusercontent.com/91064070/217234438-dbcb4473-bef4-44d6-8efb-eee9a3378c30.png)

1/ To generate the model, plot the results and print the statistics:
```shell
$> python3 linreg.py <file.csv> -p -s
```
Output with training on current dataset:

![image](https://user-images.githubusercontent.com/91064070/217247786-9e957689-64d3-4110-8025-51833817d29b.png)

2/ Then use the model to predict, for example, the price of a car given its mileage. The script will use the model generated by the [linreg.py] script and saved in the <theta.csv> file.
```shell
$> python3 prediction.py
```

Example of input/output:

![image](https://user-images.githubusercontent.com/91064070/217232883-c284289b-4775-43b1-8178-f34aa1ba1389.png)


## Dataset:

The dataset used for training is in the file [data.csv], you can generate another one using the [generate_dataset.py] script. It will generate a dataset with 2 columns. The data are based on the result of the model generated by the [linreg.py] script. (Not really happy with the results, I will improve it if needed.)

To use it, run:
```shell
$> python3 dataset_generator.py <filename> <number_of_rows>
```
