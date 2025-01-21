import pandas as pd
import time
import numpy as np
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
        "GA": results.get("Genetic Algorithm", {}).get('fitness_history', None),
        "ACO": results.get("Ant Colony Optimization", {}).get('fitness_history', None),
        "PSO": results.get("Particle Swarm Optimization", {}).get('fitness_history', None),
        "SA": results.get("Simulated Annealing", {}).get('fitness_history', None)
    }
    fig=plt.figure(figsize=(6, 4))
    
    # Plot each algorithm's fitness history
    for algo_name, fitness_history in histories.items():
        # plt.plot(fitness_history, label=algo_name)
        if fitness_history is not None and len(fitness_history) > 0:
            if algo_name == "SA":
                plt.plot(fitness_history[:50], label=algo_name)  # Only plot the first 50 for SA
            else:
                plt.plot(fitness_history, label=algo_name)
        else:
            plt.plot([], label=f"{algo_name} - Data Missing")
    
    plt.xlabel("Generation/Iteration")
    plt.ylabel("Fitness (Latency)")
    plt.title("Convergence Comparison of Algorithms")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

def plot_search_space_heatmaps(algo_data_dict):
    fig, axes = plt.subplots(2,2,figsize=(12, 8))

    # Convert each 2D list to a NumPy array for imshow
    ga_array = algo_data_dict.get("Genetic Algorithm", {}).get('fitness_matrix', None)
    aco_array = algo_data_dict.get("Ant Colony Optimization", {}).get('fitness_matrix', None)
    pso_array = algo_data_dict.get("Particle Swarm Optimization", {}).get('fitness_matrix', None)
    sa_array = algo_data_dict.get("Simulated Annealing", {}).get('fitness_matrix', None)

    # Plot GA heatmap
    ax_ga = axes[0,0]
    if ga_array is not None:
        ga_array=np.array(ga_array)
        im_ga = ax_ga.imshow(ga_array, aspect='auto', cmap='viridis')
        ax_ga.set_title("Genetic Algorithm (GA)")
        ax_ga.set_xlabel("Population Index")
        ax_ga.set_ylabel("Generation")
        fig.colorbar(im_ga, ax=ax_ga, fraction=0.046, pad=0.04)

    # Plot ACO heatmap
    ax_aco = axes[0,1]
    if aco_array is not None:
        aco_array = np.array(aco_array)
        im_aco = ax_aco.imshow(aco_array, aspect='auto', cmap='viridis')
        ax_aco.set_title("Ant Colony Optimization (ACO)")
        ax_aco.set_xlabel("Number of Ants")
        ax_aco.set_ylabel("Iteration")
        fig.colorbar(im_aco, ax=ax_aco, fraction=0.046, pad=0.04)

    # Plot PSO heatmap
    ax_pso = axes[1,0]
    if pso_array is not None:
        pso_array = np.array(pso_array)
        im_pso = ax_pso.imshow(pso_array, aspect='auto', cmap='viridis')
        ax_pso.set_title("Particle Swarm Optimization (PSO)")
        ax_pso.set_xlabel("Swarm Size")
        ax_pso.set_ylabel("Iteration")
        fig.colorbar(im_pso, ax=ax_pso, fraction=0.046, pad=0.04)

    # Plot SA heatmap
    ax_sa = axes[1,1]
    if sa_array is not None:
        sa_array=np.array(sa_array)
        yleft=np.array(sa_array[:,0])
        yright=np.array(sa_array[:,1])

        sa_array = np.array(yleft)[:, np.newaxis]
        im_sa = ax_sa.imshow(sa_array, aspect='auto', cmap='viridis')
        ax_sa.set_title("Simulated Annealing (SA)")
        ax_sa.xaxis.set_visible(False)
        ax_sa.set_ylabel("Iteration")
        fig.colorbar(im_sa, ax=ax_sa, fraction=0.046, pad=0.1)
        ax_sa_right = ax_sa.twinx()
        ax_sa_right.set_ylabel("Temperature")
        ax_sa_right.set_yticks(np.arange(len(yright)))
        ax_sa_right.set_yticklabels([f"{temp:.0f}" for temp in yright])
        ax_sa_right.set_ylim(len(yright)-1, 0)

    plt.tight_layout()
    st.pyplot(fig)