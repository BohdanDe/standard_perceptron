def SGD(w, b, dw, db, learning_rate):
    new_w = w.copy()
    new_b = b.copy()
    for index, i in enumerate(w):
        for index2, j in enumerate(i):
            new_w[index][index2] -= learning_rate * dw[index][index2]

    for index, i in enumerate(b):
        new_b[index] -= learning_rate * db[index]
    return new_w, new_b