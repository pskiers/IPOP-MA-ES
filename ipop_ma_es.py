import numpy as np
import math
import csv
from datetime import datetime
import os


_EPS = 1e-8
_MEAN_MAX = 1e32


class IPOPMAES:
    def __init__(
        self,
        mean,
        sigma,
        upper_bound,
        lower_bound,
        population_size=None,
        c1=None,
        cs=None,
        cw=None,
        dsimga=None,
        seed=42,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.dim = len(mean)
        self.population_size: int = (
            population_size if population_size is not None else 4 + math.floor(3 * math.log(self.dim))
        )

        self.y = mean
        self.sigma = sigma
        self.g = 0
        self.s: np.ndarray = np.zeros((1, self.dim))
        self.M = np.identity(self.dim)
        self.mu = self.population_size // 2
        divisor = np.sum(np.array([np.log((self.population_size + 1) / 2) - np.log(m + 1) for m in range(self.mu)]))
        self.weights = np.array(
            [
                (np.log((self.population_size + 1) / 2) - np.log(m + 1)) / divisor if m < self.mu else 0
                for m in range(self.population_size)
            ]
        )
        self.mu_eff = (np.sum(self.weights) ** 2) / np.sum(self.weights**2)
        self.expected_value_normal = np.sqrt(self.dim) * (1.0 - (1.0 / (4.0 * self.dim)) + 1.0 / (21.0 * (self.dim**2)))

        self.c1 = c1 if c1 is not None else 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cs = cs if cs is not None else (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.cw = (
            cw
            if cw is not None
            else min(
                1 - self.c1 - 1e-8,
                2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + 2 * self.mu_eff / 2),
            )
        )
        self.dsigma = (
            dsimga
            if dsimga is not None
            else 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        )

        # IPOP constants
        self.tolFun = 1e-12
        self.tolX = 1e-12
        self.conditioncov = 1e14
        self.tolXUp = 1e4

    def ask(self):
        while True:
            z = self.rng.randn(self.dim)
            d = self.M @ z
            x = self.y + self.sigma * d
            if np.all(x >= self.lower_bound) and np.all(x <= self.upper_bound):
                return x, z

    def tell(self, solutions):
        assert len(solutions) == self.population_size, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        solutions.sort(key=lambda s: s[1])
        self.max_fun = solutions[-1][1]
        self.min_fun = solutions[0][1]
        x = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        d = (x - self.y) / self.sigma  # ~ N(0, C)
        z = np.array([s[2] for s in solutions])  # ~ N(0, I)

        # Selection and recombination
        d_w = np.sum(d.T * self.weights, axis=1)
        # update mean
        self.y += self.sigma * d_w
        z_w = np.sum(z.T * self.weights, axis=1)
        # update s
        self.s = (1 - self.cs) * self.s + np.sqrt(self.mu_eff * self.cs * (2 - self.cs)) * z_w
        # update M matrix
        self.M = self.M @ (
            np.identity(self.dim)
            + (self.c1 / 2) * (self.s.T @ self.s - np.identity(self.dim))
            + (self.cw / 2)
            * (
                np.sum(
                    (z.reshape(self.population_size, self.dim, 1) @ z.reshape(self.population_size, 1, self.dim))
                    * self.weights[:, np.newaxis, np.newaxis],
                    axis=0,
                )
                - np.identity(self.dim)
            )
        )
        # update std deviations
        self.sigma *= np.exp((self.cs / self.dsigma) * ((np.linalg.norm(self.s)) / (self.expected_value_normal) - 1))

    def should_stop(self):
        # B, D = self._eigen_decomposition()
        D2, B = np.linalg.eigh(self.M)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        dC = np.diag(self.M)

        # Stop if the range of function values of the recent generation is below tolfun.
        if self.max_fun - self.min_fun < self.tolFun:
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self.sigma * dC < self.tolX) and np.all(self.sigma * self.s < self.tolX):
            return True

        # # Stop if detecting divergent behavior.
        if self.sigma * np.max(D) > self.tolXUp:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self.y == self.y + (0.2 * self.sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        if np.all(self.y == self.y + (0.1 * self.sigma * D * B)):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self.conditioncov:
            return True

        return False


def ipop_maes(fun, lbounds, ubounds, budget, seed=42):
    lower_bounds, upper_bounds = np.array(lbounds), np.array(ubounds)

    mean = lower_bounds + (np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds))
    sigma = (upper_bounds.mean() - lower_bounds.mean()) / 5  # 1/5 of the domain width
    optimizer = IPOPMAES(mean=mean, sigma=sigma, lower_bound=lower_bounds, upper_bound=upper_bounds, seed=seed)

    points = []
    while budget > 0:
        solutions = []
        for _ in range(optimizer.population_size):
            x, z = optimizer.ask()
            value = fun(x)
            solutions.append((x, value, z))
            points.append((x, value))
            budget -= 1
        optimizer.tell(solutions)

        if optimizer.should_stop():
            # popsize multiplied by 2 before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds))
            sigma = (upper_bounds.mean() - lower_bounds.mean()) / 5  # 1/5 of the domain width
            optimizer = IPOPMAES(
                mean=mean,
                sigma=sigma,
                population_size=popsize,
                lower_bound=lower_bounds,
                upper_bound=upper_bounds,
                seed=seed + popsize,  # so that we don't start with the same generations
            )
    dir = f"results/{fun.id.split('_')[1]}/{fun.id.split('_')[3]}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    csv_file = f"{dir}/IPOP_MA-ES_seed={seed}"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [f"x_{i}" for i in range(len(points[0][0]))] + ["value"]
        writer.writerow(header)
        for array, value in points:
            array_list = array.tolist()
            row = array_list + [value]
            writer.writerow(row)
    return min(points, key=lambda x: x[1])[0]
