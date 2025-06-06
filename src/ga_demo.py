import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import math


def parse_function(func_str: str):
    """Create a function from a string expression using math module."""

    def func(x: float) -> float:
        return eval(func_str, {"x": x, **math.__dict__})

def run_ga(func_str: str, x_min: float, x_max: float, pop_size: int, generations: int,
           mutation_rate: float, crossover_rate: float):
    """Simple genetic algorithm for 1D function minimization."""
    func = parse_function(func_str)
    population = np.random.uniform(x_min, x_max, pop_size)
    best_history = []

    for _ in range(generations):
        fitness = -func(population)
        best_history.append(-fitness.max())

        selected = []
        for _ in range(pop_size):
            i, j = np.random.randint(0, pop_size, 2)
            selected.append(population[i] if fitness[i] > fitness[j] else population[j])
        selected = np.array(selected)

        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                p1, p2 = selected[i], selected[(i + 1) % pop_size]
                selected[i] = alpha * p1 + (1 - alpha) * p2
                selected[(i + 1) % pop_size] = alpha * p2 + (1 - alpha) * p1

        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                selected[i] += np.random.normal(0, (x_max - x_min) * 0.1)
                selected[i] = np.clip(selected[i], x_min, x_max)

        population = selected

    fitness = -func(population)
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_value = func(best_solution)

    return best_solution, best_value, best_history


def run_button_clicked():
    try:
        expr = expr_var.get()
        x_min = float(xmin_var.get())
        x_max = float(xmax_var.get())
        pop_size = int(pop_var.get())
        generations = int(gen_var.get())
        mutation = float(mut_var.get())
        crossover = float(cross_var.get())
    except ValueError:
        result_var.set("Invalid input")
        return

    best_sol, best_val, history = run_ga(expr, x_min, x_max, pop_size, generations,
                                        mutation, crossover)
    result_var.set(f"Best solution: {best_sol:.4f} \nValue: {best_val:.4f}")

    ax.clear()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best value")
    canvas.draw()


root = tk.Tk()
root.title("Genetic Algorithm Demo")

expr_var = tk.StringVar(value="x**2 + sin(x)")
xmin_var = tk.StringVar(value="-10")
xmax_var = tk.StringVar(value="10")
pop_var = tk.StringVar(value="50")
gen_var = tk.StringVar(value="100")
mut_var = tk.StringVar(value="0.1")
cross_var = tk.StringVar(value="0.7")
result_var = tk.StringVar()

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky="nsew")

fields = [
    ("Function f(x)", expr_var),
    ("x min", xmin_var),
    ("x max", xmax_var),
    ("Population size", pop_var),
    ("Generations", gen_var),
    ("Mutation rate", mut_var),
    ("Crossover rate", cross_var),
]

for i, (label_text, var) in enumerate(fields):
    ttk.Label(frame, text=label_text).grid(row=i, column=0, sticky="w")
    ttk.Entry(frame, textvariable=var).grid(row=i, column=1, sticky="ew")

run_btn = ttk.Button(frame, text="Run GA", command=run_button_clicked)
run_btn.grid(row=len(fields), column=0, columnspan=2, pady=5)

result_label = ttk.Label(frame, textvariable=result_var)
result_label.grid(row=len(fields) + 1, column=0, columnspan=2, pady=5)

fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

root.mainloop()
