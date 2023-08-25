import numpy as np

from scipy import stats
from typing import List, Union, Optional


class ErrorTerms():
    """
    This class creates error terms to be used in the utility function

    Attributes:
    ---------------
    - distribution: either a valid scipy frozen distribution, or a list of values
    - value: current value of the error terms, which is a draw from distribution
    - dist_type: type of distribution (scipy object, list, or other)

    Methods:
    ---------------
    - draw: draws a value from distribution
    """
    def __init__(self, distribution: Union[List, type(stats.norm(0,1)), None] = None) -> None:
        """
        Parameters:
        ---------------
        - distribution: either a valid scipy frozen distribution, or a list of values
        """
        self.distribution = distribution
        self.value = None

        #type for random objects  -- change this if you're not using Scipy to handle your distributions
        self.dist_type = type(stats.norm(0,1))

    def draw(self, n_draws: int = 1) -> Union[float, np.array, List]:
        """
        Draws values from the class distirbution

        Parameters
        ---------------
        - n_draws: number of draws

        Returns
        ---------------
        Single draw or list of draws
        """

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
    """
    This class creates a normally distributed error term. The default is a standard normal distribution.

    Attributes:
    ---------------
    - mean: mean of the normal distribution
    - std: mean of the normal distribution

    Methods:
    ---------------
    - update: updates parameters of the distribution
    - draw: draws a value from distribution
    """
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        """
        Parameters:
        ---------------
        - mean: mean of the normal distribution
        - std: standard deviation of the normal distribution
        """
        super().__init__(distribution = stats.norm(loc = mean, scale = std))

        self.mean = mean
        self.std = std

    def update(self, mean: Optional[float] = None, std: Optional[float] = None) -> None:
        """
        Updates parameters of the distribution

        Parameters:
        ---------------
        - mean: new mean of the normal distribution
        - std: new standard deviation of the normal distribution
        """
        if mean:
            self.mean = mean
        if std:
            self.std = std

        self.distribution = stats.norm(loc = self.mean, scale = self.std)

    def draw(self, n_draws: int = 1)-> Union[float, np.array, List]:
        """
        Draws values from the class distirbution

        Parameters
        ---------------
        - n_draws: number of draws

        Returns
        ---------------
        Single draw or list of draws
        """
        return super().draw(n_draws)


class EVError(ErrorTerms):
    """
    This class creates an EV distributed error term. The default is a type 1 gumbel distribution.

    Attributes:
    ---------------
    - loc: location of the EV distribution
    - scale: scale of the EV distribution

    Methods:
    ---------------
    - update: updates parameters of the distribution
    - draw: draws a value from distribution
    """
    def __init__(self, loc: float = 0, scale: float = 1) -> None:
        """
        Parameters:
        ---------------
        - loc: location of the EV distribution
        - scale: scale of the EV distribution
        """
        super().__init__(distribution = stats.gumbel_r(loc = loc, scale = scale))

        self.loc = loc
        self.scale = scale

    def update(self, loc: Optional[float] = None, scale: Optional[float] = None) -> None:
        """
        Updates parameters of the distribution

        Parameters:
        ---------------
        - loc: new location of the EV distribution
        - scale: new scale of the EV distribution
        """
        if loc:
            self.loc = loc
        if scale:
            self.scale = scale

        self.distribution = stats.gumbel_r(loc = self.loc, scale = self.scale)

    def draw(self, n_draws: int = 1)-> Union[float, np.array, List]:
        """
        Draws values from the class distirbution

        Parameters
        ---------------
        - n_draws: number of draws

        Returns
        ---------------
        Single draw or list of draws
        """
        return super().draw(n_draws)


class PseudoRandomError(ErrorTerms):
    """
    This class creates pseudorandom error terms.

    Attributes:
    ---------------
    - distribution: list of pseudorandom terms

    Methods:
    ---------------
    - update: updates parameters of the distribution
    - draw: draws a value from distribution
    """
    def __init__(self, distribution: List) -> None:
        """
        Parameters:
        ---------------
        - distribution: list of pseudorandom terms
        """
        super().__init__(distribution=distribution)

    def update(self, distribution: List) -> None:
        """
        Updates pseudorandom terms.

        Parameters:
        ---------------
        - distribution: new list of pseudorandom terms
        """
        self.distribution = distribution

    def draw(self, index_draw: int = None, n_draws: int = 1) -> float:
        """
        Draws values from the class distirbution

        Parameters
        ---------------
        - index_draw: index of draw in the list of pseudorandom values
        - n_draws: number of draws

        Returns
        ---------------
        Single draw or list of draws
        """
        if index_draw and index_draw < len(self.distribution):
            self.value = self.distribution[index_draw]
        else:
            self.value = np.random.choice(self.distribution)

        return self.value
