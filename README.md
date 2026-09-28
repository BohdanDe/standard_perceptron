# standard_perceptron

Standard perceptron реалізує тренування багатошарового перцептрона зі стохастичним градієнтним спуском використовуючи переважно стандартні бібліотеки Python (окрім обробки датасету для загрузки якого було використано PIL та numpy)
Матричні множення, активаційні функції та бекпропагейшн написані на чистих списках Python
Сам проєкт є навчальним і реалізує саму математику без апаратних прискорень виконуючись на одному ядрі, отже його швидкість за замірами близько в 2500 разів менша за ідентичну нейромережу натреновану на ідентичному датасеті використовуючи PyTorch та відеокарту RTX 5060

# Особливості

```text
Математика зворотнього поширення помилки реалізована без сторонніх бібліотек
Модульна архітектура, модель створюється подібно до бібліотеки Keras (Приклад - Sequential([Dense(size=128, input_size=784, activation_function='relu'),)
Логування з виводом точност після кожної епохи
Динамічна зміна темпу навчання
Випадкова перетасовка датасету для запобіганнч заучуванню
Оптимізатор SGD
Функція активації ReLU та Softmax
```

# підготовка

Для обробки датасету було використано дві бібліотеки
pip install numpy Pillow

## загрузка датасету

```text
project_root/
│
├── mnist/
│   ├── 0/ (зображення нулів)
│   ├── 1/ (зображення одиниць)
│   └── ...
├── standard_perceptron.py
├── train.py
├── optimizers.py
├── losses.py
```

Назви зображень не грають ролі але повинні мати розширення .png або .jpg

## запуск

python train.py

# архітектура нейромережі

Вхідний шар - Вектор розміром 784 (flattened зобрадення 28х28 пікселів)
Прихований шар - Dense(size=128, activation='relu')
Вихідний шар - Dense(size=10, activation='softmax')

# результати

```text
Результати тренування на 20 епохах на 1000 зображеннях (10 на клас)

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

Було досягнуто точність в 88% за 20 епох
```
