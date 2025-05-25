from deap_er import algorithms


def evolve(toolbox, pop, pc, pm, num_gen):
    pop_size = len(pop)

    for _ in range(0, num_gen):
        for ind in pop:
            if not ind.fitness.is_valid():
                ind.fitness.values = toolbox.evaluate(ind)

        offspring = toolbox.select(pop, pop_size)
        pop = algorithms.var_and(toolbox, offspring, pc, pm)

    for ind in pop:
            if not ind.fitness.is_valid():
                ind.fitness.values = toolbox.evaluate(ind)
    return pop
