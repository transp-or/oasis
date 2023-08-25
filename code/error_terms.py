import numpy as np

from scipy import stats
from typing import List, Union, Optional


class ErrorTerms():
    def __init__(self, distribution: Union[List, type(stats.norm(0,1)), None] = None) -> None:
        self.distribution = distribution
        self.value = None

        #type for random objects  -- change this if you're not using Scipy to handle your distributions
        self.dist_type = type(stats.norm(0,1))

    def draw(self, n_draws: int = 1) -> Union[float, np.array, List]:
        if isinstance(self.distribution, self.dist_type):
            self.value = self.distribution.rvs(size = n_draws)
            if n_draws == 1:
                self.value = self.value[0]
        elif isinstance(self.distribution, List):
            self.value = np.random.choice(self.distribution, size = n_draws)

        else:
            print('Please provide a valid distribution (scipy frozen object or list of values)')

        return self.value


class GaussianError(ErrorTerms):
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        super().__init__(distribution = stats.norm(loc = mean, scale = std))

        self.mean = mean
        self.std = std

    def update(self, mean: Optional[float] = None, std: Optional[float] = None) -> None:
        if mean:
            self.mean = mean
        if std:
            self.std = std

        self.distribution = stats.norm(loc = self.mean, scale = self.std)

    def draw(self, n_draws: int = 1)-> Union[float, np.array, List]:
        return super().draw(n_draws)


class EVError(ErrorTerms):
    def __init__(self, loc: float = 0, scale: float = 1) -> None:
        super().__init__(distribution = stats.gumbel_r(loc = loc, scale = scale))

        self.loc = loc
        self.scale = scale

    def update(self, loc: Optional[float] = None, scale: Optional[float] = None) -> None:
        if loc:
            self.loc = loc
        if scale:
            self.scale = scale

        self.distribution = stats.gumbel_r(loc = self.loc, scale = self.scale)

    def draw(self, n_draws: int = 1)-> Union[float, np.array, List]:
        return super().draw(n_draws)


class PseudoRandomError(ErrorTerms):
    def __init__(self, distribution: List) -> None:
        super().__init__(distribution=distribution)

    def update(self, distribution: List) -> None:
        self.distribution = distribution

    def draw(self, index_draw: int = None, n_draws: int = 1) -> float:
        if index_draw and index_draw < len(self.distribution):
            self.value = self.distribution[index_draw]
        else:
            self.value = np.random.choice(self.distribution)

        return self.value
