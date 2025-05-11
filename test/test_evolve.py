import random

import pytest
from deap import base, creator, tools

from evolve import evolve


@pytest.fixture
def creator_types():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    yield
    del creator.Individual
    del creator.FitnessMax

@pytest.fixture
def toolbox(creator_types):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)                                
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return sum(ind),
    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

    return toolbox

def test_evolve_basic(creator_types, toolbox):
    pop = toolbox.population(n=1000)
    pop = evolve(toolbox, pop, 0.75, 0.15, 20)

    assert pop[0].fitness.values[0] > 30