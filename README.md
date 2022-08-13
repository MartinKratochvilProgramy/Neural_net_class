# Neural net class
Neural net class abstracts most of the boilerplate code required to create neural net using keras library. By default code uses 'adam' optimizer and ReLU activation function.
# How to use
Clone the repo and include class in your project.

```python
nn = Neural_net(10000, (32, 32), 9604) 
nn.load_data('input.csv', 0.8)
nn.train_model(150_000, 1., 'he_whole')
nn.save_model('he_whole')
```