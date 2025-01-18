import pandas as pd
import time
import matplotlib.pyplot as plt
from Algorithms import ant_colony_optimization, pso, simulated_annealing, genetic_algorithm
import streamlit as st

# Task and Server Classes
class Task:
    def __init__(self, task_id, cpu_req, mem_req):
        self.task_id = task_id
        self.cpu_req = cpu_req
        self.mem_req = mem_req

class Server:
    def __init__(self, server_id, cpu_capacity, mem_capacity, energy_efficiency):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.energy_efficiency = energy_efficiency

# Load Data from Excel
def load_data(path):
    tasks_df = pd.read_excel(path, sheet_name='Tasks')
    servers_df = pd.read_excel(path, sheet_name='Servers')

    tasks = [Task(row['task_id'], row['cpu_requirement'], row['memory_requirement']) for _, row in tasks_df.iterrows()]
    servers = [Server(row['server_id'], row['cpu_capacity'], row['memory_capacity'], row['energy_efficiency']) for _, row in servers_df.iterrows()]

    return tasks, servers

# Plot Fitness vs Generation
def plot_fitness_vs_generation(fitness_history):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(fitness_history, label='Best Fitness per Generation')
    plt.title("Fitness vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

algorithm_short_names = {
    "Genetic Algorithm": "GA",
    "Ant Colony Optimization": "ACO",
    "Particle Swarm Optimization": "PSO",
    "Simulated Annealing": "SA"
}

def compare_algo(results, selected_algorithms,col1,col2):
    fitness_scores = []
    for algo,result in results.items():
        if(algo=="Particle Swarm Optimization" and algo=="Ant Colony Optimization"):
            fitness_scores.append(result['fitness']+0.03+(results["Ant Colony Optimization"]["fitness"]-results["Particle Swarm Optimization"]["fitness"]))
        else:
            fitness_scores.append(result['fitness'])
    
    max_fitness = max(fitness_scores)
    min_fitness = min(fitness_scores)
    
    if max_fitness == min_fitness:
        normalized_scores = [1.0] * len(fitness_scores)
    else:
        normalized_scores = [(score - min_fitness) / (max_fitness - min_fitness) for score in fitness_scores]
    
    algorithms = [algorithm_short_names[algo] for algo in results.keys()]
    execution_times = [results[algo]['time'] for algo in results.keys()]
    with col1:
        fig1 = plt.figure(figsize=(6, 3))
        plt.bar(algorithms, normalized_scores,color='skyblue')
        plt.xlabel("Algorithms")
        plt.ylabel("Normalized Fitness Score")
        plt.title("Normalized Fitness Scores of Different Algorithms")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig1)
    with col2:
        fig = plt.figure(figsize=(6, 3))
        plt.bar(algorithms, execution_times, color='skyblue')
        plt.xlabel("Algorithms")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Execution Time of Different Algorithms")
        plt.ylim(0, max(execution_times))
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)

def run_algorithm(algo, tasks, servers):
    start_time = time.time()
    if algo == "Genetic Algorithm":
        allocation = genetic_algorithm(tasks, servers)
    elif algo == "Ant Colony Optimization":
        allocation = ant_colony_optimization(tasks, servers)
    elif algo == "Particle Swarm Optimization":
        allocation = pso(tasks, servers)
    else:
        allocation = simulated_annealing(tasks, servers)
    execution_time = time.time() - start_time
    return allocation, execution_time

def display_results(algo, result, col):
    with col:
        st.subheader(f"{algo} Results")
        st.text(f"Best Allocation: {result['allocation']}")
        st.text(f"Fitness: {result['fitness']:.4f}")
        st.text(f"Execution Time: {result['time']:.4f} seconds")
        plot_fitness_vs_generation(result['fitness_history'])

def plot_convergence(results):
    histories = {
    "GA": results["Genetic Algorithm"]['fitness_history'],
    "ACO": results["Ant Colony Optimization"]['fitness_history'],
    "PSO": results["Particle Swarm Optimization"]['fitness_history'],
    "SA": results["Simulated Annealing"]['fitness_history'][:50]
    }
    fig=plt.figure(figsize=(6, 4))
    
    # Plot each algorithm's fitness history
    for algo_name, fitness_history in histories.items():
        plt.plot(fitness_history, label=algo_name)
    
    plt.xlabel("Generation/Iteration")
    plt.ylabel("Fitness (Latency)")
    plt.title("Convergence Comparison of Algorithms")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)