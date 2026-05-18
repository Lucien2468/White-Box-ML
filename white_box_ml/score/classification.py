import numpy as np


class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        predictions = np.argmax(predictions.data, axis=1)
        labels = np.argmax(labels.data, axis=1)
        self.correct += np.sum(predictions == labels)
        self.total += len(labels)

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0


class Precision:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0

    def update(self, predictions, labels):
        predictions = np.argmax(predictions.data, axis=1)
        labels = np.argmax(labels.data, axis=1)
        self.true_positives += np.sum((predictions == 1) & (labels == 1))
        self.false_positives += np.sum((predictions == 1) & (labels == 0))

    def compute(self):
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
