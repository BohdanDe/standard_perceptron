def transpose(x):
    matrix = x

    rows = len(matrix)
    cols = len(matrix[0])

    transpose = []

    for i in range(cols):
        new_row = []
        for j in range(rows):
            new_row.append(matrix[j][i])
        transpose.append(new_row)
    return transpose

def softmax_cross_entropy_gradient(y_pred, y_true):
    # e g [0.708, 0.292] - [1, 0] = [-0.292, 0.292]
    loss = []
    for i in range(len(y_true)):
        loss.append(y_pred[i] - y_true[i])
    return loss