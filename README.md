# Neural net class
Neural net class abstracts most of the boilerplate code required to create neural net using keras library. By default code uses 'adam' optimizer, ReLU activation function and mean-squared error - this can be changed in the __init__ method.
# How to use
Clone the repo and include class in your project.
Create neural net with 8 input neurons, 2 output neurons and two layer with 32 and 16 neurons:
```python
nn = Neural_net(8, (32, 16), 2) 
```
Load data from csv file. Csv file has to be structured according to the neural net structure, in this case with 8 inputs and 2 outputs each row has to have 10 numbers, first 8 corresponding to the input vector, last two corresponding to the input vector - example input with two datapoints can be seen in 'input_example.csv'. Second argument is the fraction in which to split data into training and validation datasets, in this case 0.8 means 80% of data will be used for training. Data is shuffled. Use None to not split dataset.
```python
nn.load_data('input.csv', 0.8)
```
Train data for 150 epochs and end training if mean-squared error is less than 1%. Training plot will be saved as 'convergence(32,16).jpg':
```python
nn.train_model(150, 1., 'convergence')
```
Save model with name 'neural_net_8(32, 16)2'
```python
nn.save_model('neural_net')
```
Load neural net:
```python
nn.load_model('model name')
```
Calculate prediction using custom input vector, returns predicted vector:
```python
prediction = nn.predict(input_vector)
```
# Create and train your models in a couple of lines of code
```python
nn = Neural_net(8, (32, 16), 2) 
nn.load_data('input.csv', 0.8)
nn.train_model(150, 1., 'convergence')
nn.save_model('neural_net')
nn.load_model('model name')
prediction = nn.predict(input_vector)
```
