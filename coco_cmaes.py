from cmaes import CMA
import numpy as np
import os
import csv


def ipop_cmaes(fun, lbounds, ubounds, budget, seed=42):
    lower_bounds, upper_bounds = np.array(lbounds), np.array(ubounds)
    bounds = np.array([lower_bounds, upper_bounds]).T

    mean = lower_bounds + (np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds))
    sigma = (upper_bounds.mean() - lower_bounds.mean()) / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=seed)

    points = []
    while budget > 0:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = fun(x)
            solutions.append((x, value))
            points.append((x, value))
            budget -= 1
        optimizer.tell(solutions)

        if optimizer.should_stop():
            # popsize multiplied by 2 before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds))
            sigma = (upper_bounds.mean() - lower_bounds.mean()) / 5  # 1/5 of the domain width
            optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize, bounds=bounds, seed=seed + popsize)
    dir = f"results/{fun.id.split('_')[1]}/{fun.id.split('_')[3]}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    csv_file = f"{dir}/IPOP_CMA-ES_seed={seed}"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [f"x_{i}" for i in range(len(points[0][0]))] + ["value"]
        writer.writerow(header)
        for array, value in points:
            array_list = array.tolist()
            row = array_list + [value]
            writer.writerow(row)
    return min(points, key=lambda x: x[1])[0]

def cmaes(fun, lbounds, ubounds, budget, seed=42):
    lower_bounds, upper_bounds = np.array(lbounds), np.array(ubounds)
    bounds = np.array([lower_bounds, upper_bounds]).T

    mean = lower_bounds + (np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds))
    sigma = (upper_bounds.mean() - lower_bounds.mean()) / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=seed)

    points = []
    while budget > 0:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = fun(x)
            solutions.append((x, value))
            points.append((x, value))
            budget -= 1
        optimizer.tell(solutions)
    dir = f"results/{fun.id.split('_')[1]}/{fun.id.split('_')[3]}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    csv_file = f"{dir}/CMA-ES_seed={seed}"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [f"x_{i}" for i in range(len(points[0][0]))] + ["value"]
        writer.writerow(header)
        for array, value in points:
            array_list = array.tolist()
            row = array_list + [value]
            writer.writerow(row)
    return min(points, key=lambda x: x[1])[0]
