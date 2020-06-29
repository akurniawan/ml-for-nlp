import numpy as np


class MultinomialNaiveBayesTextClassifier(object):
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.prior = None
        self.counts_by_c = np.array([0])
        self.likelihood = None
        self.total_vocabularies = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 1

        total_data = x.shape[0]
        self.total_vocabularies = np.unique(x).shape[0]
        self.prior = np.array([len(x[y == c]) / total_data for c in np.unique(y)])
        # Adding default count for unknown vocabulary
        self.counts_by_c = np.array(
            [
                np.append(
                    np.array([0]),
                    np.histogram(x[y == c], bins=self.total_vocabularies)[0],
                )
                for c in np.unique(y)
            ]
        )
        # Remove vocab with frequencies less than self.min_freq
        self.counts_by_c[self.counts_by_c < self.min_freq] = 0
        # Add smoothing
        self.counts_by_c += 1
        total_words_by_c = (self.counts_by_c - 1).sum(axis=0)
        total_words_by_c[total_words_by_c == 0] = 1
        self.likelihood = np.array(
            [words / total_words_by_c for words in self.counts_by_c]
        )
        return self

    def predict_proba(self, x: np.array):
        X = np.copy(x)
        # Handling unknown words that are not in our vocabularies yet
        X[X > self.total_vocabularies] = 0
        probs = np.zeros((X.shape[0], self.prior.shape[0]))
        for idx, row in enumerate(X):
            curr_likelihood = self.likelihood[:, row]
            log_likelihood = np.log(curr_likelihood).sum(axis=1) + np.log(self.prior)
            likelihood = np.exp(log_likelihood)
            probs[idx] = likelihood
        return probs
