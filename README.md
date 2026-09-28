# standard_perceptron

Standard perceptron implements the training of a multilayer perceptron with stochastic gradient descent using mostly standard Python libraries (except for dataset processing, for the loading PIL and numpy were used)

Matrix multiplications, activation functions and backpropagation are written in pure Python lists

The project itself is educational and implements the math itself without hardware acceleration running on a single core, therefore its speed according to measurements is about 2500 times lower than an identical neural network trained on an identical dataset using PyTorch and an RTX 5060 video card

# Features

- Backpropagation math is implemented without third-party libraries
- Modular architecture, the model is created similarly to the Keras library (Example - Sequential([Dense(size=128, input_size=784, activation_function='relu'),)
- Logging with accuracy output after each epoch
- Dynamic learning rate change
- Random dataset shuffling to prevent memorization
- SGD optimizer
- ReLU and Softmax activation functions

# preparation

Two libraries were used to process the dataset
pip install numpy Pillow

## dataset loading

```text
project_root/
│
├── mnist/
│   ├── 0/ (images of zeros)
│   ├── 1/ (images of ones)
│   └── ...
├── standard_perceptron.py
├── train.py
├── optimizers.py
├── losses.py
```

The names of the images do not matter but must have the .png or .jpg extension

## launch

python train.py

# neural network architecture

Input layer - Vector of size 784 (flattened 28x28 pixel image)
Hidden layer - Dense(size=128, activation='relu')
Output layer - Dense(size=10, activation='softmax')

# results

Training results on 20 epochs on 1000 images (10 per class)

```text
Layer 0, params: 100480, activation: relu
Layer 1, params: 1290, activation: softmax
Total params: 101770
Loaded 1000 images.
First image size: 784 pixels
First image size: 784 pixels
First label: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Pre-training evaluation's accuracy: 0.032
Epoch № 0 Accuracy: 0.632 Learning rate: 0.001
Epoch № 1 Accuracy: 0.736 Learning rate: 0.001
Epoch № 2 Accuracy: 0.788 Learning rate: 0.001
Epoch № 3 Accuracy: 0.824 Learning rate: 0.001
Epoch № 4 Accuracy: 0.84 Learning rate: 0.001
Epoch № 5 Accuracy: 0.844 Learning rate: 0.001
Epoch № 6 Accuracy: 0.848 Learning rate: 0.001
Epoch № 7 Accuracy: 0.852 Learning rate: 0.001
Epoch № 8 Accuracy: 0.856 Learning rate: 0.001
Epoch № 9 Accuracy: 0.856 Learning rate: 0.001
Epoch № 10 Accuracy: 0.868 Learning rate: 0.0007
Epoch № 11 Accuracy: 0.868 Learning rate: 0.0007
Epoch № 12 Accuracy: 0.872 Learning rate: 0.0007
Epoch № 13 Accuracy: 0.876 Learning rate: 0.0007
Epoch № 14 Accuracy: 0.876 Learning rate: 0.0007
Epoch № 15 Accuracy: 0.876 Learning rate: 0.0005
Epoch № 16 Accuracy: 0.876 Learning rate: 0.0005
Epoch № 17 Accuracy: 0.876 Learning rate: 0.0005
Epoch № 18 Accuracy: 0.876 Learning rate: 0.0005
Epoch № 19 Accuracy: 0.88 Learning rate: 0.0005
```

An accuracy of 88% was achieved in 20 epochs
