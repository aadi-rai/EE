from copy import deepcopy

import numpy as np
from deap_er import algorithms, tools


def evolve(toolbox, pop, pc, pm, num_elitism, num_gen):
    stats = tools.Statistics(key=lambda ind: sum(ind.fitness.wvalues))
    stats.register("max", np.max)

    pareto_front = tools.ParetoFront()
    logbook = tools.Logbook()

    for gen in range(0, num_gen):
        selection = toolbox.select(pop, len(pop) - num_elitism)
        pop = algorithms.var_and(toolbox, selection, pc, pm) + tools.sel_best(pop, num_elitism)

        for ind in pop:
            if not ind.fitness.is_valid():
                ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(pop)

        pareto_front.update(pop)
        record["pareto_front"] = deepcopy(pareto_front)

        logbook.record(gen=gen, **record)
    return pop, logbook