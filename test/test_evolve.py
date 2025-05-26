import random

import pytest
from deap_er import base, creator, tools

from evolve import evolve


@pytest.fixture
def fitness_max():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    yield
    del creator.Individual
    del creator.FitnessMax

@pytest.fixture
def toolbox_single(fitness_max):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_float, size=10)                                
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)

    def evaluate(ind):
        return sum(ind),
    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_gaussian, mu=0, sigma=1, mut_prob=0.1)

    return toolbox

def test_evolve_single(fitness_max, toolbox_single):
    pop = toolbox_single.population(size=1000)
    pop, logbook = evolve(toolbox_single, pop, 0.75, 0.15, 20)

    assert pop[0].fitness.values[0] > 30

@pytest.fixture
def fitness_multi():
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,-1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    yield
    del creator.Individual
    del creator.FitnessMulti

@pytest.fixture
def toolbox_multi(fitness_multi):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.init_repeat, creator.Individual, toolbox.attr_float, size=10)                                
    toolbox.register("population", tools.init_repeat, list, toolbox.individual)

    def evaluate(ind):
        # https://pymoo.org/problems/multi/zdt.html
        f1 = ind[0]
        g = 1 + sum(ind[1:])
        f2 = 1 - abs(f1 / g) ** 0.5

        return f1, f2
    
    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.sel_tournament, contestants=3)
    toolbox.register("mate", tools.cx_two_point)
    toolbox.register("mutate", tools.mut_gaussian, mu=0, sigma=1, mut_prob=0.1)

    return toolbox

def test_evolve_multi(fitness_multi, toolbox_multi):
    pop = toolbox_multi.population(size=1000)
    pop, logbook = evolve(toolbox_multi, pop, 0.75, 0.15, 20)

    assert sum(pop[0].fitness.wvalues) > 5