import math
from collections import Counter, defaultdict

# Categorical features, hashable labels; positive alpha, nonempty training data.
class NaiveBayes:
    def __init__(self, nclass=None, alpha=1.0):
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = alpha

    def fit(self, X, y):
        # Count class priors and feature values conditioned on each class.
        self.count_y = Counter(y)
        self.count_x_y = defaultdict(lambda: defaultdict(Counter))
        self.vocab = [set() for _ in X[0]]
        for row, label in zip(X, y):
            for i, value in enumerate(row):
                self.vocab[i].add(value)
                self.count_x_y[label][i][value] += 1
        return self

    def predict(self, X):
        # For each row: log prior + sum of log conditional likelihoods.
        # Choose the label with the largest score; no normalization needed.
        labels = []
        total = sum(self.count_y.values())
        for row in X:
            scores = {}
            for label, count in self.count_y.items():
                score = math.log(count / total)
                for i, value in enumerate(row):
                    # Reserve one unknown category per feature.
                    denominator = count + self.alpha * (len(self.vocab[i]) + 1)
                    numerator = self.count_x_y[label][i][value] + self.alpha
                    score += math.log(numerator / denominator)
                scores[label] = score
            labels.append(max(scores, key=scores.get))
        return labels
