import numpy as np
import sys, math, random


def binary_pso(iter, particle_size, pop_size):
    t = 0
    w = 0.5
    c1 = 1
    c2 = 1
    r1 = round(random.uniform(0, 1.1), 1)
    r2 = round(random.uniform(0, 1.1), 1)
    speed_boundary = [-0.6, 0.6]

    pop = initialize(particle_size, pop_size)
    p_best = pop
    print("Population:")
    print(pop)
    
    best_fitness = [fitness(particle) for particle in pop]
    
    g_best = p_best[best_fitness.index(max(best_fitness))]
    print("gBest:")
    print(g_best)
    old_gBest_fitness = 0
    gBest_fitness = fitness(g_best)

    speed = np.zeros(shape=(particle_size, pop_size))
    print("Speed:")
    print(speed)
    
    while t < iter and old_gBest_fitness < gBest_fitness:
        t += 1
        old_gBest_fitness = gBest_fitness
        print("t: ", end="")
        print(t)
        speed = up_speed(speed, w, c1, r1, c2, r2, pop, p_best, g_best, speed_boundary)
        print("Speed:")
        print(np.array(speed))

        pop = up_pop(speed)
        print("Populasi:")
        print(pop)
        
        p_best, best_fitness = up_p_best(pop, p_best, best_fitness)
        print(p_best)
        
        g_best = p_best[best_fitness.index(max(best_fitness))]
        gBest_fitness = fitness(g_best)
        print(g_best)

def binary_to_decimal(particle):
    particle = particle[::-1]
    num = 0
    for i, bit in enumerate(particle):
        num += bit * 2 ** i
    return num

def fitness(particle):
    # -x2 + 14x – 13
    print("Particle:")
    print(particle, end="")
    x = binary_to_decimal(particle)
    print(- x ** 2 + 14 * x - 13)
    return - x ** 2 + 14 * x - 13

def initialize(particle_size, pop_size):
    population = []
    for i in range(pop_size):
        population.append(np.random.choice([0, 1], size=particle_size))
    
    return np.array(population)

def up_speed(speed, w, c1, r1, c2, r2, pop, p_best, g_best, speed_boundary):
    # w.v + c1.r1(pBest - x) + c2.r2(gBest - x)
    new_speed = []
    for i, particle in enumerate(pop):
        v = []
        for j, bit in enumerate(particle):
            # print(w, speed[i][j], c1, r1, p_best[i][j], bit, c2, r2, g_best[j], bit)
            v.append(w * speed[i][j] + 
            c1 * r1 * (p_best[i][j] - bit) + c2 * r2 * (g_best[j] - bit))
        new_speed.append(v)
    
    new_speed = np.array(new_speed)
    new_speed = np.where(new_speed < speed_boundary[0], speed_boundary[0], new_speed)
    new_speed = np.where(new_speed > speed_boundary[1], speed_boundary[1], new_speed)
    
    return np.array(new_speed)

def sig_function(v):
    # sig = 1 / (1 + e^-v)
    return 1 / (1 + math.exp(-v))

def compare(sig, rand_value):
    return int(rand_value < sig)

def up_pop(speed):
    vec_sig = np.vectorize(sig_function)
    sig = vec_sig(speed)
    print("Sig:")
    print(sig)
    random_array = np.random.random(speed.shape)
    print("rand[0,1]")
    print(random_array)

    vec_compare = np.vectorize(compare)

    return vec_compare(sig, random_array)

def up_p_best(pop, p_best, best_fitness):
    pop_fitness = [fitness(particle) for particle in pop]
    for i in range(len(pop)):
        p_best[i] = p_best[i] if best_fitness[i] > pop_fitness[i] else pop[i]
    
    return (p_best, [fitness(particle) for particle in p_best])


if __name__ == '__main__':
    binary_pso(10, 4, 4)
