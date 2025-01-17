import streamlit as st
from Helper import load_data, plot_fitness_vs_generation, compare_algo,display_results,run_algorithm

st.set_page_config(layout="wide")
st.title("Edge Server Load Balancing")

filename = st.file_uploader("Choose Dataset File", type=["xlsx", "xls"],help="OR Use Default File")

algorithms = [
    "Genetic Algorithm",
    "Ant Colony Optimization",
    "Particle Swarm Optimization",
    "Simulated Annealing"
]
selected_algorithms = st.multiselect("Select Optimization Algorithms", algorithms, default=algorithms)

tasks = None
servers = None

if filename is None:
    filename = "Dataset.xlsx"

if filename is not None:
    try:
        tasks, servers = load_data(filename)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

if st.button("Run Selected Algorithms") and tasks is not None and servers is not None:
    results = {}
    col1, col2 = st.columns(2)
    
    for algo in selected_algorithms:
        allocation, exec_time = run_algorithm(algo, tasks, servers)
        results[algo] = {
            'fitness_history': allocation['fitness history'],
            'allocation': allocation['best_individual'],
            'fitness': allocation['fitness'],
            'time': exec_time
        }
    
    for i, (algo, result) in enumerate(results.items()):
        col = col1 if i % 2 == 0 else col2
        display_results(algo, result, col)
    
    compare_algo(results, selected_algorithms)
