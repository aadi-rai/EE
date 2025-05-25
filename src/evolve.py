from deap import algorithms


def evolve(toolbox, pop, pc, pm, num_gen):
    pop_size = len(pop)

    for _ in range(0, num_gen):
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        offspring = toolbox.select(pop, pop_size)
        pop = algorithms.varAnd(offspring, toolbox, pc, pm)

    for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
    return pop

