## Linear Regression: Predicting the price of a car given its kilometerage


![image](https://user-images.githubusercontent.com/91064070/213888620-6b488e8f-0164-4655-a442-af6fe28b1bb3.png)

![image](https://user-images.githubusercontent.com/91064070/213887705-0a1ac769-e042-4d45-a0c6-717af65efeea.png)


## Usage:

1/ To generate the model and plot the results:
```shell
$> python3 model.py
```
Output with training on current dataset:

![image](https://user-images.githubusercontent.com/91064070/216848511-613c5023-874b-4fba-8a4e-8fbfe14eea35.png)

2/ Then use the model to predict the price of a car given its mileage:
```shell
$> python3 price_prediction.py
```

Example of input/output:

![image](https://user-images.githubusercontent.com/91064070/216846997-d7bf4d3e-584c-416f-9f22-1bdd31298c6f.png)


## Dataset:

The dataset used for training is in the file [data.csv], you can generate another one using the [generate_dataset.py] script. It will generate a dataset with 50 rows and 2 columns, this time with houses surface as independant variable and their price as dependant variable.

To use it, run:
```shell
$> python3 dataset_generator.py
```

You will have to modify the [model.py] file to use the new dataset. Replace 'km' by 'm2' in the [model.py] file, and replace 'data.csv'  by 'houses_prices.csv'.