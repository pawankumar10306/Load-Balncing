import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st


def calculate_latency(task, server):
    cpu_latency = task.cpu_req / server.cpu_capacity
    mem_latency = task.mem_req / server.mem_capacity
    total_latency = cpu_latency + mem_latency
    return total_latency

def generate_latency_matrix(tasks, servers):
    num_tasks = len(tasks)
    num_servers = len(servers)
    latency_matrix = np.zeros((num_tasks, num_servers))

    for i, task in enumerate(tasks):
        for j, server in enumerate(servers):
            latency_matrix[i, j] = calculate_latency(task, server)  # Latency in milliseconds
    return latency_matrix

def best_task_server_assignment(latency_matrix, tasks, servers):
    best_assignments = []
    total_latency = 0

    for i in range(len(tasks)):
        # Find the server with the minimum latency for the current task
        min_latency_index = np.argmin(latency_matrix[i, :])
        best_server = servers[min_latency_index]
        best_assignments.append(best_server.server_id)  # Add server ID to the assignment list
        total_latency += latency_matrix[i, min_latency_index]  # Add the latency

    return best_assignments, total_latency

def plot_latency_matrix(latency_matrix, tasks, servers):
    plt.figure(figsize=(8, 4))
    plt.imshow(latency_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Latency (ms)')

    # Add server and task labels
    plt.xticks(ticks=np.arange(len(servers)), labels=[s.server_id for s in servers], rotation=45)
    plt.yticks(ticks=np.arange(len(tasks)), labels=[t.task_id for t in tasks])
    plt.xlabel("Servers")
    plt.ylabel("Tasks")
    plt.title("Latency Matrix (Task vs Server)")

    # Highlight the minimum value in each row
    for i in range(latency_matrix.shape[0]):  # Iterate over rows
        min_value_index = np.argmin(latency_matrix[i, :])  # Index of minimum value in the row
        plt.scatter(min_value_index, i, color='red', label='Min Latency' if i == 0 else "")

    # Prevent duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_labels = {label: handle for handle, label in zip(handles, labels)}
    plt.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()




# Function to plot Fitness vs Individual
def plot_fitness_vs_individual(population, fitness_values, generation):
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(population)), fitness_values, c='blue', label='Fitness per Individual')
    plt.title(f"Fitness vs Individual (Generation {generation})")
    plt.xlabel("Individual")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.show()


# Create a random individual (task-server assignment)
def create_individual(tasks, servers):
    return [random.choice(servers) for _ in tasks]

# Calculate the fitness (minimize latency across all task-server assignments)
def fitness(individual, tasks):
    total_latency = 0
    for task, server in zip(tasks, individual):
        total_latency += calculate_latency(task, server)
    return total_latency

# Create the initial population
def create_population(pop_size, tasks, servers):
    return [create_individual(tasks, servers) for _ in range(pop_size)]

# Select parents using tournament selection
def tournament_selection(population, tasks, tournament_size):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda ind: fitness(ind, tasks))  # Sort by fitness (lower is better)
    return tournament[0], tournament[1]  # Return the top two individuals

# Crossover between two parents (swap random segments of the task-server assignment)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation (randomly reassign a task to a different server)
def mutate(individual, servers, mutation_rate):
    return [
        random.choice(servers) if random.random() < mutation_rate else server
        for server in individual
    ]



# Main genetic algorithm
def genetic_algorithm(tasks, servers):
    # Parameters
    pop_size = 100  # Population size
    generations = 50  # Number of generations
    mutation_rate = 0.01  # Probability of mutation
    tournament_size = 3  # Size of the tournament for selection
    n_best = 10  # Number of best individuals to keep in the next generation

    # Step 1: Create initial population
    population = create_population(pop_size, tasks, servers)
    fitness_history = []

    for generation in range(generations):
        # Step 2: Evaluate fitness of the population
        population.sort(key=lambda ind: fitness(ind, tasks))  # Sort by fitness (lower is better)
        fitness_history.append(fitness(population[0], tasks))

        # Step 3: Plot Fitness vs Individual for this generation
        fitness_values = [fitness(ind, tasks) for ind in population]

        # print(f"\nGeneration {generation+1}: Best fitness (latency) = {fitness(population[0], tasks):.4f}")
        # plot_fitness_vs_individual(population, fitness_values, generation)

        # Step 4: Create the next generation
        next_generation = population[:n_best]

        # Select parents and create children
        while len(next_generation) < pop_size:
            parent1, parent2 = tournament_selection(population[n_best:], tasks, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1, servers, mutation_rate), mutate(child2, servers, mutation_rate)])

        # Step 5: Replace the old population with the new one
        population = next_generation[:pop_size]

    # Plot fitness vs generation
    # print("\n\nFitness vs Generation:")
    # plot_fitness_vs_generation(fitness_history)

    # Return the best solution with its fitness
    best_individual = population[0]
    assignment = [server.server_id for task, server in zip(tasks, best_individual)]
    return {
        "fitness history":fitness_history,
        "best_individual": assignment,
        "fitness": fitness(best_individual, tasks)
    }

# tasks, servers = load_data()

# Generate latency matrix
# latency_matrix = generate_latency_matrix(tasks, servers)
# plot_latency_matrix(latency_matrix, tasks, servers)

# Manually Done
# Best_Assignments, Best_Latency = best_task_server_assignment(latency_matrix, tasks, servers)
# print(f"Best solution (task-server assignment): {Best_Assignments}")
# print(f"Fitness of the best solution: {Best_Latency:.4f}")

# Run the genetic algorithm
# GeneticAlgorithm = genetic_algorithm(tasks, servers)
# print(f"Best solution (task-server assignment): {GeneticAlgorithm['best_individual']}")
# print(f"Fitness of the best solution: {GeneticAlgorithm['fitness']:.4f}")


# Simulated Annealing
def compute_total_latency_sa(solution, latency_matrix):
    indices = (np.arange(len(solution)), solution)
    total_latency = np.sum(latency_matrix[indices])
    return total_latency

def simulated_annealing(tasks, servers):
    num_tasks = len(tasks)
    num_servers = len(servers)

    # SA Parameters
    initial_temperature = 100.0
    final_temperature = 0.1
    cooling_rate = 0.95
    max_iterations = 100

    # Generate an initial random solution
    latency_matrix =  generate_latency_matrix(tasks, servers)
    current_solution = np.random.randint(0, num_servers, size=num_tasks)
    current_fitness = compute_total_latency_sa(current_solution, latency_matrix)

    best_solution = np.copy(current_solution)
    best_fitness = current_fitness

    fitness_history = [best_fitness]

    temperature = initial_temperature

    for iteration in range(max_iterations):
        if temperature <= final_temperature:
            break

        # Generate a neighbor solution by randomly reassigning one task to a different server
        neighbor_solution = np.copy(current_solution)
        random_task = np.random.randint(0, num_tasks)
        random_server = np.random.randint(0, num_servers)
        neighbor_solution[random_task] = random_server

        neighbor_fitness = compute_total_latency_sa(neighbor_solution, latency_matrix)

        # Calculate the acceptance probability
        delta = neighbor_fitness - current_fitness
        if delta < 0:
            # Better solution found, accept it
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness
        else:
            # Accept worse solution with a probability
            acceptance_prob = np.exp(-delta / temperature)
            if np.random.rand() < acceptance_prob:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness

        # Update the best solution if needed
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = np.copy(current_solution)

        # Store the best fitness in the history for plotting
        fitness_history.append(best_fitness)

        # Decrease the temperature
        temperature *= cooling_rate

        # print(f"Iteration {iteration}: Best fitness (latency) = {best_fitness:.4f}")

    # Plot fitness vs iteration
    # print("\nOptimization complete. Final Best Fitness =", best_fitness)
    # plot_fitness_vs_generation(fitness_history)

    return {
        "fitness history":fitness_history,
        "best_individual": [servers[idx].server_id for idx in best_solution],
        "fitness": best_fitness
    }

# Simulated Annealing
# Simulated_solution = simulated_annealing(tasks, servers, latency_matrix)
# print("Best solution (task-server assignment):", Simulated_solution['best_individual'])
# print(f"Fitness of the best solution: {Simulated_solution['fitness']:.4f}")


# Function for ants to construct solutions
def construct_solution(tasks, servers, pheromone_matrix, eta_matrix, alpha, beta):
    num_tasks = len(tasks)
    num_servers = len(servers)
    solution = []
    for i in range(num_tasks):  # For each task
        # Compute the probability of assigning task i to server j
        probabilities = []
        for j in range(num_servers):
            tau_ij = pheromone_matrix[i][j]
            eta_ij = eta_matrix[i][j]
            probability = (tau_ij ** alpha) * (eta_ij ** beta)
            probabilities.append(probability)
        probabilities = np.array(probabilities)
        # Normalize the probabilities
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum == 0:
            # If all probabilities are zero, assign equal probability
            probabilities = np.ones(num_servers) / num_servers
        else:
            probabilities /= probabilities_sum
        # Choose a server based on the probabilities
        server_index = np.random.choice(num_servers, p=probabilities)
        solution.append(server_index)
    return solution

# Function to compute total latency of a solution
def compute_total_latency_aco(solution, tasks, servers):
    total_latency = 0
    for task, server_index in zip(tasks, solution):
        server = servers[server_index]
        total_latency += calculate_latency(task, server)
    return total_latency

# Function to update pheromones
def update_pheromones(pheromone_matrix, solutions, fitnesses, rho, Q):
    num_tasks, num_servers = pheromone_matrix.shape

    # Evaporate pheromones
    pheromone_matrix *= (1 - rho)

    # Deposit pheromones
    for solution, fitness in zip(solutions, fitnesses):
        delta_pheromone = Q / fitness
        for i, server_index in enumerate(solution):
            pheromone_matrix[i][server_index] += delta_pheromone
    return pheromone_matrix

# Run ACO
def ant_colony_optimization(tasks, servers):
    # Parameters
    num_ants = 30  # Number of ants
    iterations = 50  # Number of iterations
    alpha = 1.0  # Influence of pheromone
    beta = 2.0  # Influence of heuristic information
    rho = 0.1  # Pheromone evaporation rate
    Q = 1  # Constant for pheromone update

    num_tasks = len(tasks)
    num_servers = len(servers)

    # Step 1: Initialize pheromone levels
    pheromone_matrix = np.ones((num_tasks, num_servers))

    # Step 2: Compute heuristic information (eta)
    # Use inverse of latency matrix
    latency_matrix = generate_latency_matrix(tasks, servers)
    eta_matrix = 1.0 / latency_matrix
    # Replace infinite values with zero (if any latency is zero, which would cause infinite eta)
    eta_matrix[np.isinf(eta_matrix)] = 0

    # Keep track of the best solution
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for iteration in range(iterations):
        solutions = []
        fitnesses = []

        # Each ant constructs a solution
        for ant in range(num_ants):
            solution = construct_solution(tasks, servers, pheromone_matrix, eta_matrix, alpha, beta)
            total_latency = compute_total_latency_aco(solution, tasks, servers)
            solutions.append(solution)
            fitnesses.append(total_latency)

            # Update the best solution found so far
            if total_latency < best_fitness:
                best_fitness = total_latency
                best_solution = solution

        # Update pheromones
        pheromone_matrix = update_pheromones(pheromone_matrix, solutions, fitnesses, rho, Q)

        fitness_history.append(best_fitness)

        # Optionally, print intermediate results
        # print(f"Iteration {iteration+1}: Best fitness (latency) = {best_fitness:.4f}")

    # Plot fitness vs iteration
    # print("\n\nFitness vs Iteration:")
    # plot_fitness_vs_generation(fitness_history)

    # Return the best solution with its fitness
    assignment = [servers[server_index].server_id for server_index in best_solution]
    return {
        "fitness history":fitness_history,
        "best_individual": assignment,
        "fitness": best_fitness
    }


# Run the Ant Colony Optimization algorithm
# ACO_solution = ant_colony_optimization(tasks, servers)
# print(f"Best solution (task-server assignment): {ACO_solution['best_individual']}")
# print(f"Fitness of the best solution: {ACO_solution['fitness']:.4f}")



# Particle Swarm Optimization Algorithm
def pso(tasks, servers):
    # Parameters for PSO
    swarm_size = 100
    iterations = 50
    w = 0.7    # Inertia weight
    c1 = 1.5   # Cognitive component
    c2 = 1.5   # Social component

    num_tasks = len(tasks)
    num_servers = len(servers)

    # Initialize population (positions & velocities)
    positions = np.random.uniform(0, num_servers-1, (swarm_size, num_tasks))
    velocities = np.random.uniform(-1, 1, (swarm_size, num_tasks))

    def discretize_position(pos):
        # Map position to nearest integer server index within range
        int_pos = np.round(pos).astype(int)
        int_pos = np.clip(int_pos, 0, num_servers-1)
        return int_pos

    # Evaluate initial population
    fitnesses = []
    for i in range(swarm_size):
        assignment = discretize_position(positions[i])
        f = compute_total_latency_aco(assignment, tasks, servers)
        fitnesses.append(f)
    fitnesses = np.array(fitnesses)

    # Initialize personal bests
    pbest_positions = np.copy(positions)
    pbest_fitnesses = np.copy(fitnesses)

    # Initialize global best
    gbest_index = np.argmin(pbest_fitnesses)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_fitness = pbest_fitnesses[gbest_index]

    fitness_history = [gbest_fitness]

    # Main PSO loop
    for iteration in range(iterations):
        for i in range(swarm_size):
            r1 = np.random.random(num_tasks)
            r2 = np.random.random(num_tasks)

            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (pbest_positions[i] - positions[i])
                             + c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]

            # Evaluate fitness after update
            assignment = discretize_position(positions[i])
            f = compute_total_latency_aco(assignment, tasks, servers)

            # Update personal best if better
            if f < pbest_fitnesses[i]:
                pbest_fitnesses[i] = f
                pbest_positions[i] = np.copy(positions[i])

            # Update global best if better
            if f < gbest_fitness:
                gbest_fitness = f
                gbest_position = np.copy(positions[i])

        fitness_history.append(gbest_fitness)
        # print(f"Iteration {iteration+1}: Best fitness (latency) = {gbest_fitness:.4f}")

    # Plot the fitness over iterations
    # print("\nFitness vs Iteration:")
    # plot_fitness_vs_generation(fitness_history)

    best_assignment = discretize_position(gbest_position)
    return {
        "fitness history":fitness_history,
        "best_individual": [servers[idx].server_id for idx in best_assignment],
        "fitness": gbest_fitness
    }

# Run PSO
# PSO_solution = pso(tasks, servers)
# print(f"Best solution (task-server assignment): {PSO_solution['best_individual']}")
# print(f"Fitness of the best solution: {PSO_solution['fitness']:.4f}")

# Manually Done
# print(latency_matrix)
# print(f"\nBest solution (Manually Calculated): {Best_Assignments}")
# print(f"Fitness of the best solution: {Best_Latency:.4f}")

#Genetic Algorithm
# print(f"Best solution (Genetic Algorithm): {GeneticAlgorithm['best_individual']}")
# print(f"Fitness of the best solution: {GeneticAlgorithm['fitness']:.4f}")

# Ant Colony Optimization
# print(f"Best solution (task-server assignment): {ACO_solution['best_individual']}")
# print(f"Fitness of the best solution: {ACO_solution['fitness']:.4f}")

# Particle Swarm Optimization
# print(f"Best solution (task-server assignment): {PSO_solution['best_individual']}")
# print(f"Fitness of the best solution: {PSO_solution['fitness']:.4f}")

# Simulated Annealing
# print("Best solution (task-server assignment):", Simulated_solution['best_individual'])
# print(f"Fitness of the best solution: {Simulated_solution['fitness']:.4f}")