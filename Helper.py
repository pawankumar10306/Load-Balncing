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
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history, label='Best Fitness per Generation')
    plt.title("Fitness vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.legend()
    st.pyplot()

def compare_algo(results,selected_algorithms):
    fitness_scores = [result['fitness'] for result in results.values()]
    max_fitness = max(fitness_scores)
    min_fitness = min(fitness_scores)
    plt.figure(figsize=(10, 2))
    plt.bar(selected_algorithms, fitness_scores)
    plt.xlabel("Algorithms")
    plt.ylabel("Fitness Score")
    plt.title("Fitness Scores of Different Algorithms")
    plt.ylim(min_fitness - 1, max_fitness + 1)
    st.pyplot()

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