import numpy as np


class HMM(object):
    def __init__(self, num_vocabs: int, num_tags: int):
        self._num_tags = num_tags
        self.transition = np.random.uniform(size=(num_tags + 1, num_tags + 1))
        self.observation = np.random.uniform(size=(num_vocabs, num_tags + 1))

    def fit(self, x, y):
        alphas = self._alphas(x)
        betas = self._betas(x)

    def _betas(self, x):
        betas = np.empty((len(x), self._num_tags))
        beta = np.ones((1, self._num_tags))
        for step in reversed(range(1, len(x))):
            beta = (
                beta
                * self.transition[: self._num_tags, : self._num_tags]
                * self.observation[x[len(x) - 1], : self._num_tags]
            ).sum(0)
            betas[step] = beta
        beta = (
            beta * self.transition[self._num_tags, : self._num_tags] * self.observation[x[len(x) - 1], : self._num_tags]
        ).sum(0)
        betas[0] = beta
        return betas

    def _alphas(self, x):
        alphas = np.empty((len(x), self._num_tags))
        alpha = self.transition[self._num_tags, : self._num_tags] * self.observation[x[0], : self._num_tags]
        alphas[0] = alpha
        for step in range(1, len(x)):
            alpha = (
                alpha
                * self.transition[: self._num_tags, : self._num_tags]
                * self.observation[x[step], : self._num_tags]
            ).sum(0)
            alphas[step] = alpha
        return alphas

    def predict(self, x):
        pass


if __name__ == "__main__":
    hmm = HMM(5, 2)
    hmm.fit(np.array([3, 1, 3], dtype=np.int), np.array([]))
