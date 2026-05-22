from flask import Blueprint, request, render_template
import numpy as np
from sympy import symbols, lambdify, parse_expr, latex


views = Blueprint('views', __name__)

@views.route('/')
def main():
    return render_template("main.html")

# -------------------------------
# Algoritmos metaheurísticos
# -------------------------------

def GA(objf, lb, ub, dim, population, generations, mutation_rate, crossover_rate):
    # Aseguramos un mínimo de población para que el algoritmo funcione
    if population < 4: population = 4 
    
    pop = np.random.uniform(lb, ub, (population, dim))
    fitness = np.array([objf(ind) for ind in pop])
    history = []
    
    for g in range(generations):
        # Aseguramos que siempre haya al menos 2 padres para el cruce
        n_parents = max(2, population // 2)
        parents_idx = np.argsort(fitness)[:n_parents]
        parents = pop[parents_idx]
        children = []
        
        while len(children) < (population - len(parents)):
            # Selección de 2 padres aleatorios
            idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
            p1, p2 = parents[idx1], parents[idx2]
            
            # Crossover
            if np.random.rand() < crossover_rate and dim > 1:
                cut = np.random.randint(1, dim)
                child = np.concatenate([p1[:cut], p2[cut:]])
            else:
                # Si es 1D o no hay crossover, elegimos uno de los dos padres
                child = p1.copy() if np.random.rand() < 0.5 else p2.copy()
            
            # Mutación
            if np.random.rand() < mutation_rate:
                m_idx = np.random.randint(dim)
                child[m_idx] = np.random.uniform(lb, ub)
            
            children.append(child)
        
        pop = np.vstack([parents, np.array(children)])
        fitness = np.array([objf(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        history.append({
            "generation": g, 
            "solution": pop[best_idx].tolist(), 
            "fitness": float(fitness[best_idx])
        })
        
    best_idx = np.argmin(fitness)
    return pop[best_idx].tolist(), float(fitness[best_idx]), history
    
def PSO(objf, lb, ub, dim, particles, iterations, c1, c2, w):
    X = np.random.uniform(lb, ub, (particles, dim))
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_val = np.array([objf(x) for x in X])
    gbest = pbest[np.argmin(pbest_val)]
    history = []
    for t in range(iterations):
        for i in range(particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(gbest-X[i])
            X[i] += V[i]
            X[i] = np.clip(X[i], lb, ub)
            val = objf(X[i])
            if val < pbest_val[i]:
                pbest[i], pbest_val[i] = X[i].copy(), val
        gbest = pbest[np.argmin(pbest_val)]
        history.append({"generation": t, "solution": gbest.tolist(), "fitness": float(objf(gbest))})
    return gbest.tolist(), float(objf(gbest)), history

def ACO(objf, lb, ub, dim, ants, alpha, beta, evaporation, iterations):
    pheromone = np.ones((dim,))
    best_sol, best_val = None, float("inf")
    history = []
    for t in range(iterations):
        for _ in range(ants):
            sol = np.random.uniform(lb, ub, dim)
            val = objf(sol)
            if best_sol is None or val < best_val:
                best_sol, best_val = sol, val
        pheromone = (1-evaporation)*pheromone + alpha*np.random.rand(dim)
        history.append({"generation": t, "solution": best_sol.tolist(), "fitness": float(best_val)})
    return best_sol.tolist(), float(best_val), history

def AIS(objf, lb, ub, dim, antibodies, cloning_rate, alpha, beta, iterations):
    pop = np.random.uniform(lb, ub, (antibodies, dim))
    fitness = np.array([objf(ind) for ind in pop])
    best_sol, best_val = pop[np.argmin(fitness)], np.min(fitness)
    history = []
    for t in range(iterations):
        clones = []
        for i in range(antibodies):
            for _ in range(int(cloning_rate)):
                clone = pop[i] + np.random.normal(0, alpha, dim)
                clone = np.clip(clone, lb, ub)
                clones.append(clone)
        clones = np.array(clones)
        clone_fit = np.array([objf(c) for c in clones])
        combined = np.vstack([pop, clones])
        combined_fit = np.concatenate([fitness, clone_fit])
        idx = np.argsort(combined_fit)[:antibodies]
        pop, fitness = combined[idx], combined_fit[idx]
        if fitness[0] < best_val:
            best_sol, best_val = pop[0], fitness[0]
        history.append({"generation": t, "solution": best_sol.tolist(), "fitness": float(best_val)})
    return best_sol.tolist(), float(best_val), history

def DE(objf, lb, ub, dim, population, mutation_factor, crossover_rate, iterations):
    pop = np.random.uniform(lb, ub, (population, dim))
    fitness = np.array([objf(ind) for ind in pop])
    best_sol, best_val = pop[np.argmin(fitness)], np.min(fitness)
    history = []
    for t in range(iterations):
        for i in range(population):
            idxs = [idx for idx in range(population) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mutation_factor*(b-c)
            cross_points = np.random.rand(dim) < crossover_rate
            trial = np.where(cross_points, mutant, pop[i])
            trial = np.clip(trial, lb, ub)
            val = objf(trial)
            if val < fitness[i]:
                pop[i], fitness[i] = trial, val
        best_sol, best_val = pop[np.argmin(fitness)], np.min(fitness)
        history.append({"generation": t, "solution": best_sol.tolist(), "fitness": float(best_val)})
    return best_sol.tolist(), float(best_val), history

# -------------------------------
# Ruta principal
# -------------------------------
@views.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    try:
        data = request.form
        problem_type = data.get('problem_type', 'function')
        algo = data.get('algorithm')
        plot_data = None
        params = {}
        
        # --- PHASE 1: PREPARE THE PROBLEM ---
        # --- PHASE 1: PREPARE THE PROBLEM ---
        if problem_type == 'tsp':
            raw_cities = data.get('tsp_cities', '').strip().split('\n')
            cities = []
            
            # Get bounds for validation (defaulting to 0-100 if not specified for TSP)
            # Or use the lb/ub inputs from the form
            limit_lb = float(data.get('lb', 0))
            limit_ub = float(data.get('ub', 100))

            for line_no, line in enumerate(raw_cities, 1):
                line = line.strip()
                if not line: continue  # Skip empty lines
                
                try:
                    parts = line.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"Formato incorrecto en línea {line_no}: debe ser x,y")
                    
                    x_c, y_c = map(float, parts)

                    # --- RANGE VALIDATION ---
                    if not (limit_lb <= x_c <= limit_ub) or not (limit_lb <= y_c <= limit_ub):
                        return render_template("main.html", 
                            error=f"Ciudad en línea {line_no} ({x_c}, {y_c}) fuera de rango [{limit_lb}, {limit_ub}].")
                    
                    cities.append([x_c, y_c])
                except ValueError as e:
                    return render_template("main.html", error=f"Error en datos de TSP: {str(e)}")

            if len(cities) < 3:
                return render_template("main.html", error="El TSP requiere al menos 3 ciudades para formar un ciclo.")
            
            dim = len(cities)
            # Important: The algorithm explores a 'priority' space (0 to 1)
            # the cities are just used for distance calculation.
            lb, ub = 0, 1 
            mode = "min"
            expr_str = "Problema del Viajero"

            def objf(vec):
                # The 'vec' contains values between 0 and 1. 
                # argsort turns these into a sequence of city indices.
                route = np.argsort(vec)
                dist = 0
                for i in range(len(route)):
                    c1 = cities[route[i]]
                    c2 = cities[route[(i + 1) % len(route)]]
                    dist += np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                return dist
            
            plot_data = {"type": "tsp", "cities": cities}

        else:
            expr_str = data.get('function', 'x**2')
            expr_str = expr_str.replace('^', '**')
            expr_str = expr_str.replace(' ', '')
            dim = int(data.get('dim', 1))
            lb = float(data.get('lb', -5))
            ub = float(data.get('ub', 5))
            mode = data.get('optimization_type', 'min')

            x, y = symbols('x y')
            
            # Usamos transformations para que Sympy sea más flexible con la entrada
            from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
            transformations = (standard_transformations + (implicit_multiplication_application,))
            
            if dim == 1:
                # Nos aseguramos de que y sea 0 si la función solo es 1D
                expr = parse_expr(expr_str, transformations=transformations).subs(y, 0)
                var_tuple = (x,)
            elif dim == 2:
                expr = parse_expr(expr_str, transformations=transformations)
                var_tuple = (x, y)
            else:
                expr = parse_expr(expr_str)
                var_tuple = symbols(f'x0:{dim}')

            f_sym = lambdify(var_tuple, expr, modules=["numpy", "math"])

            f_sym = lambdify(var_tuple, expr, modules=["numpy", "math"])

            def f_original(vec):
                try:
                    args = list(vec)
                    res = f_sym(args[0]) if dim == 1 else f_sym(*args)
                    
                    # Si res es un objeto de Sympy (como zoo o nan), evalf() fallará o dará error
                    final_val = float(res.evalf()) if hasattr(res, 'evalf') else float(res)
                    
                    # Si el resultado no es un número finito, lanzamos error manual
                    if not np.isfinite(final_val):
                        return 1e18 # Un valor de fitness muy alto (penalización)
                    return final_val
                except (ZeroDivisionError, OverflowError, TypeError):
                    return 1e18 # Penalización por caer en una zona inválida (división por cero)

            objf = (lambda v: -f_original(v)) if mode == "max" else f_original

            # Plot Data Generation
            if dim == 1:
                xs = np.linspace(lb, ub, 200)
                ys = [f_original([xi]) for xi in xs]
                plot_data = {"type": "function", "x": xs.tolist(), "y": ys}
            elif dim == 2:
                xs = np.linspace(lb, ub, 50)
                ys = np.linspace(lb, ub, 50)
                X, Y = np.meshgrid(xs, ys)
                Z = np.array([[f_original([X[i,j], Y[i,j]]) for j in range(50)] for i in range(50)])
                plot_data = {"type": "function", "x": xs.tolist(), "y": ys.tolist(), "z": Z.tolist()}

        # --- PHASE 2: CONFIGURE ALGO & RUN ---
        if algo == 'GA':
            params = {
                "Población": int(data.get('ga_population', 50)),
                "Generaciones": int(data.get('ga_generations', 100)),
                "Tasa Mutación": float(data.get('ga_mutation', 0.05)),
                "Tasa Crossover": float(data.get('ga_crossover', 0.8))
            }
            best_sol, best_val, history = GA(objf, lb, ub, dim, 
                                             params["Población"], params["Generaciones"], 
                                             params["Tasa Mutación"], params["Tasa Crossover"])
        elif algo == 'PSO':
            params = {
                "Partículas": int(data.get('pso_particles', 50)),
                "Iteraciones": int(data.get('pso_iterations', 100)),
                "C1": float(data.get('pso_c1', 1.5)),
                "C2": float(data.get('pso_c2', 1.5)),
                "W (Inercia)": float(data.get('pso_w', 0.9))
            }
            best_sol, best_val, history = PSO(objf, lb, ub, dim, 
                                              params["Partículas"], params["Iteraciones"], 
                                              params["C1"], params["C2"], params["W (Inercia)"])
        elif algo == 'ACO':
            params = {
                "Hormigas": int(data.get('aco_ants', 50)),
                "Iteraciones": int(data.get('aco_iterations', 100)),
                "Alpha": float(data.get('aco_alpha', 1)),
                "Beta": float(data.get('aco_beta', 2)),
                "Evaporación": float(data.get('aco_evaporation', 0.3))
            }
            best_sol, best_val, history = ACO(objf, lb, ub, dim, 
                                              params["Hormigas"], params["Alpha"], 
                                              params["Beta"], params["Evaporación"], params["Iteraciones"])
        elif algo == 'AIS':
            params = {
                "Anticuerpos": int(data.get('ais_antibodies', 100)),
                "Iteraciones": int(data.get('ais_iterations', 100)),
                "Tasa Clonación": float(data.get('ais_cloning', 3)),
                "Alpha": float(data.get('ais_alpha', 2)),
                "Beta": float(data.get('ais_beta', 1))
            }
            best_sol, best_val, history = AIS(objf, lb, ub, dim, 
                                              params["Anticuerpos"], params["Tasa Clonación"], 
                                              params["Alpha"], params["Beta"], params["Iteraciones"])
        elif algo == 'DE':
            params = {
                "Población": int(data.get('de_population', 60)),
                "Iteraciones": int(data.get('de_iterations', 100)),
                "Factor Mutación": float(data.get('de_mutation', 0.9)),
                "Tasa Crossover": float(data.get('de_crossover', 0.5))
            }
            best_sol, best_val, history = DE(objf, lb, ub, dim, 
                                             params["Población"], params["Factor Mutación"], 
                                             params["Tasa Crossover"], params["Iteraciones"])
        else:
            return render_template("main.html", error="Algoritmo no soportado")

        # --- PHASE 3: FINAL TOUCHES ---
        if mode == "max":
            best_val = -best_val
            for h in history: h["fitness"] = -h["fitness"]

        # Convert the mathematical expression to a beautiful LaTeX format
        if problem_type == 'tsp':
            latex_func = "Problema del Viajero"
        else:
            latex_func = f"f(x) = {latex(expr)}" if dim == 1 else f"f(x, y) = {latex(expr)}"

        return render_template(
            'result.html',
            algorithm=algo,
            best_solution=list(best_sol),
            best_value=float(best_val),
            history=history,
            function=latex_func, # <--- Pass the new LaTeX string here
            dim=dim,
            plot_data=plot_data,
            params=params,
            mode=mode
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template("main.html", error=f"Error Crítico: {str(e)}")