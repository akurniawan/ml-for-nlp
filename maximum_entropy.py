import numpy as np


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def softmax(gx, w, x):
    shift = gx - np.max(gx)
    exp = np.exp(shift)
    z = exp.sum(-1)
    return exp / z[:, :, np.newaxis]


def cross_entropy(true, pred):
    log_pred = np.log(pred)
    loss = log_pred * true
    return -loss.sum(-1)


class MaxEntClassifier(object):
    def __init__(self, num_features: int, num_outputs: int, epochs: int = 10000, lr: float = 0.1):
        self.weights = np.random.uniform(-1.0, 1.0, size=(num_features, num_outputs))
        self.epochs = epochs
        self.lr = lr

    def fit(self, x: np.ndarray, y: np.ndarray):
        for i in range(self.epochs):
            projection = x.dot(self.weights)
            prediction = softmax(projection, self.weights, x)
            loss = cross_entropy(y, prediction)
            print(loss.sum() / (x.shape[0] * x.shape[1]))
            # Calculate gradients
            grad = (prediction - y)[:, :, :, np.newaxis] * x[:, :, np.newaxis, :]
            grad = np.transpose(grad, (0, 1, 3, 2))  # need to tranpose to make the dimension similar to weights
            grad = grad.sum(0) / (x.shape[0] * x.shape[1])  # accumulate all gradients from all batches
            for step in reversed(range(grad.shape[0])):
                # grad = grad.sum(1)  # accumulate all gradients from all steps
                # Apply gradient descent
                self.weights = self.weights - (grad[step] * self.lr)
        return self

    def predict(self, x: np.ndarray):
        projection = x.dot(self.weights)
        prediction = softmax(projection, self.weights, x)
        return prediction


if __name__ == "__main__":
    classifier = MaxEntClassifier(10, 5)
    x = np.random.randint(low=0, high=2, size=(2, 2, 10))
    y = np.array([[[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]])
    classifier.fit(x, y)
    print(classifier.predict(x))
