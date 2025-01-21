import streamlit as st
from Helper import load_data, plot_fitness_vs_generation, compare_algo,display_results,run_algorithm,plot_convergence,plot_search_space_heatmaps

st.set_page_config(layout="wide")
st.title("Meta-Heuristic Approaches for Resource Allocation in Edge Computing")

filename = st.file_uploader("Choose Dataset File", type=["xlsx", "xls"],help="OR Use Default File")

algorithms = [
    "Genetic Algorithm",
    "Ant Colony Optimization",
    "Particle Swarm Optimization",
    "Simulated Annealing"
]
selected_algorithms = st.multiselect(
    "Select Optimization Algorithms",
    ["Select All"] + algorithms,
    default=["Select All"]
)

if "Select All" in selected_algorithms:
    selected_algorithms = algorithms

tasks = None
servers = None

if filename is None:
    filename = "Dataset.xlsx"
    # filename = "data.xlsx"

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
            'fitness_matrix':allocation['fitness_matrix'],
            'fitness_history': allocation['fitness history'],
            'allocation': allocation['best_individual'],
            'fitness': allocation['fitness'],
            'time': exec_time
        }
    
    for i, (algo, result) in enumerate(results.items()):
        col = col1 if i % 2 == 0 else col2
        display_results(algo, result, col)

    st.divider()
    st.header("Comparison of Optimization Algorithms",divider=True)
    col1, col2 = st.columns(2)
    if len(selected_algorithms) > 1:
        compare_algo(results, selected_algorithms,col1,col2)
    st.subheader("Search Space Exploration")
    plot_search_space_heatmaps(results)
    with col1:
        plot_convergence(results)
    with col2:
        with st.expander("Show Scalability"):
            st.image("Scalability.jpg")