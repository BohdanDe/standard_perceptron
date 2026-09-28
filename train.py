from standard_perceptron import *
import os
import numpy as np
from PIL import Image

model = Sequential([Dense(size=128, input_size=784, activation_function='relu'),
                     Dense(size=10, activation_function='softmax')])
model.summary()
print(np.min(model.layers[0].weights))
print(np.max(model.layers[0].weights))
def load_mnist_custom(root_dir):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    for label in range(10):
        folder_path = os.path.join(root_dir, str(label))

        if not os.path.exists(folder_path):
            continue

        one_hot = [0] * 10
        one_hot[label] = 1

        all_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        random.shuffle(all_files)

        selected_files = all_files[:125]

        for i, filename in enumerate(selected_files):
            img_path = os.path.join(folder_path, filename)

            img = Image.open(img_path).convert('L')
            img_array = np.array(img).flatten()

            normalized_img = (img_array / 255.0).tolist()

            if i < 100:
                x_train_list.append(normalized_img)
                y_train_list.append(one_hot)
            else:
                x_test_list.append(normalized_img)
                y_test_list.append(one_hot)

    train_zipped = list(zip(x_train_list, y_train_list))
    test_zipped = list(zip(x_test_list, y_test_list))

    random.shuffle(train_zipped)
    random.shuffle(test_zipped)

    x_train, y_train = zip(*train_zipped)
    x_test, y_test = zip(*test_zipped)

    return list(x_train), list(y_train), list(x_test), list(y_test)

x_train, y_train, x_test, y_test = load_mnist_custom('mnist')
print(f"Loaded {len(x_train)} images.")
print(f"First image size: {len(x_train[0])} pixels")
print(f"First label: {y_train[0]}")

def eval(_model):
    for i in range(1):
        num_pred = 0
        num_correct = 0
        for j in range(len(x_test)):
            num_pred += 1
            r = _model.predict(x_test[j])
            if np.argmax(r) == np.argmax(y_test[j]):
                num_correct += 1
        print(f"Pre-training evaluation's accuracy: {num_correct / num_pred}")
eval(model)
for e in range(20): #epochs
    num_pred = 0
    num_correct = 0
    lr = 0.001 if e < 10 else (0.001 * 0.7 if e < 15 else 0.001 * 0.5)
    for i in range(len(x_train)):
        model.step(x_train[i], y_train[i], lr)

    for index, i in enumerate(x_test):
        num_pred += 1
        r = model.predict(i)
        if np.argmax(r) == np.argmax(y_test[index]):
            num_correct += 1

    print(f"Epoch № {e} Accuracy: {num_correct / num_pred} Learning rate: {lr}")
