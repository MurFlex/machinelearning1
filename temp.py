import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier

nb_classes = 10

(XI_train, yI_train), (XI_test, yI_test) = mnist.load_data()
X_train = XI_train
X_test = XI_test
NI_test = yI_test.size
M = 100
X_train = X_train[::M]
X_test = X_test[::M]
y_train = yI_train[::M]
y_test = yI_test[::M]

for i in range(10):
    fig, axes = plt.subplots(1, 1, figsize=(2, 2))
    x = X_train[i]
    plt.imshow(x, interpolation="none")
    
N_train = np.size(X_train, 0)
N_test = np.size(X_test, 0)

# X_train = X_train.reshape(60000, 784)
X_train = X_train.reshape(N_train, 784)
# X_test = X+test.reshape(10000, 784)
X_test = X_test.reshape(N_test, 784)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

X_train /= 255
X_test /= 255

mlp = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(40, 10),
    # activation = 'relu'
    activation='logistic',
    random_state=1)

mlp.fit(X_train, y_train)
print("Правильность на обучающем наборе: ", format(mlp.score(X_train, y_train)))
print("Правильность на тестовом наборе: ", format(mlp.score(X_test, y_test)))