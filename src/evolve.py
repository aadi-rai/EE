import random


def evolve(toolbox, pop, pc, pm, num_gen):
    for _ in range(0, num_gen):
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, len(pop))]

        for i in range(0, len(offspring), 2):
            if random.random() < pc:
                offspring[i + 1], offspring[i] = toolbox.mate(offspring[i + 1], offspring[i])
                del offspring[i + 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < pm:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        pop[:] = offspring

    for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
    return pop
