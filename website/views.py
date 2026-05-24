from flask import Blueprint, request, render_template
import numpy as np
from sympy import symbols, lambdify, parse_expr, latex


views = Blueprint('views', __name__)

@views.route('/')
def main():
    return render_template("main.html.j2")

# -------------------------------
# Algoritmos metaheurísticos
# -------------------------------

def GA(objf, lb, ub, dim, population, generations, mutation_rate, crossover_rate):
    if population < 4: population = 4 
    pop = np.random.uniform(lb, ub, (population, dim))
        
    fitness = np.array([objf(ind) for ind in pop])
    history = []
    
    for g in range(generations):
        n_parents = max(2, population // 2)
        parents_idx = np.argsort(fitness)[:n_parents]
        parents = pop[parents_idx]
        children = []
        
        while len(children) < (population - len(parents)):
            idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
            p1, p2 = parents[idx1], parents[idx2]
            
            # Crossover
            if np.random.rand() < crossover_rate and dim > 1:
                cut = np.random.randint(1, dim)
                child = np.concatenate([p1[:cut], p2[cut:]])
            else:
                child = p1.copy() if np.random.rand() < 0.5 else p2.copy()
            
            # Mutación
            if np.random.rand() < mutation_rate:
                m_idx = np.random.randint(dim)
                child[m_idx] = np.random.uniform(lb, ub)
            
            children.append(child)
        
        pop = np.vstack([parents, np.array(children)])
        fitness = np.array([objf(ind) for ind in pop])
        
        current_best = float(np.min(fitness))
        best_idx = np.argmin(fitness)
        history.append({
            "generation": g, 
            "solution": pop[best_idx].tolist(), 
            "fitness": float(fitness[best_idx]),
            "current_fitness": current_best
        })

    best_idx = np.argmin(fitness)
    return pop[best_idx].tolist(), float(fitness[best_idx]), history
    
def PSO(objf, lb, ub, dim, particles, iterations, c1, c2, w):
    X = np.random.uniform(lb, ub, (particles, dim))
        
    V = np.zeros((particles, dim), dtype=float)
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
                
        current_best = float(np.min([objf(x) for x in X]))
        gbest = pbest[np.argmin(pbest_val)]
        history.append({
            "generation": t, 
            "solution": gbest.tolist(), 
            "fitness": float(objf(gbest)),
            "current_fitness": current_best
        })
    return gbest.tolist(), float(objf(gbest)), history

def ACO(objf, lb, ub, dim, ants, alpha, beta, evaporation, iterations):
    pheromone = np.ones((dim,))
    best_sol, best_val = None, float("inf")
    history = []
    for t in range(iterations):
        gen_best_val = float("inf") 
        for _ in range(ants):
            sol = np.random.uniform(lb, ub, dim)
                
            val = objf(sol)
            if val < gen_best_val: 
                gen_best_val = val
            
            if val < best_val:
                best_sol, best_val = sol.copy(), val

        history.append({
            "generation": t, 
            "solution": best_sol.tolist(), 
            "fitness": float(best_val),
            "current_fitness": float(gen_best_val)
        })
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
        
        current_best = float(np.min(clone_fit))
        
        idx = np.argsort(combined_fit)[:antibodies]
        pop, fitness = combined[idx], combined_fit[idx]
        
        if fitness[0] < best_val:
            best_sol, best_val = pop[0].copy(), fitness[0]

        history.append({
            "generation": t, 
            "solution": best_sol.tolist(), 
            "fitness": float(best_val),
            "current_fitness": current_best
        })
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

        current_gen_min = float(np.min(fitness))
        
        if current_gen_min < best_val:
            best_val = current_gen_min
            best_sol = pop[np.argmin(fitness)].copy()

        history.append({
            "generation": t, 
            "solution": best_sol.tolist(), 
            "fitness": float(best_val),
            "current_fitness": current_gen_min 
        })
    return best_sol.tolist(), float(best_val), history


def MFO(objf, lb, ub, dim, N, Max_iter, b=1, selection_mode="Adaptativo", representation="continuous"):
    if representation == "binary":
        moths = np.random.randint(2, size=(N, dim))
    else:
        moths = np.random.uniform(lb, ub, (N, dim))
       
    fitness = np.array([objf(m) for m in moths])
    sorted_idx = np.argsort(fitness)
    flames = moths[sorted_idx].copy()
    flame_fitness = fitness[sorted_idx].copy()
    best_flame_pos = flames[0].copy()
    best_flame_score = flame_fitness[0]
    history = []

    for t in range(Max_iter):
        flame_no = int(np.ceil(N - t * ((N - 1) / Max_iter)))
        a = -1 + t * ((-1) / Max_iter)

        for i in range(N):
            if selection_mode == "Mejor Flama":
                flame = flames[0]
            elif selection_mode == "Adaptativo":
                flame = flames[i%flame_no]
            else:
                flame = flames[flame_no-1]
            distance = np.abs(flame - moths[i])
            rand_t = (a - 1) * np.random.rand() + 1
            new_pos = distance * np.exp(b * rand_t) * np.cos(2 * np.pi * rand_t) + flame
           
            if representation == "binary":
                sigmoid = 1 / (1 + np.exp(-np.clip(new_pos, -10, 10)))
                moths[i] = (np.random.rand(dim) < sigmoid).astype(int)

            else:
                moths[i] = new_pos

        if representation != "binary":
            moths = np.clip(moths, lb, ub)

        fitness = np.array([objf(m) for m in moths])
        current_best_val = float(np.min(fitness))
        combined_moths = np.vstack([flames, moths])
        combined_fitness = np.concatenate([flame_fitness, fitness])
        sorted_indices = np.argsort(combined_fitness)
        flames = combined_moths[sorted_indices[:N]]
        flame_fitness = combined_fitness[sorted_indices[:N]]

        if flame_fitness[0] < best_flame_score:
            best_flame_score = float(flame_fitness[0])
            best_flame_pos = flames[0].copy()

        history.append({
            "generation": t,
            "solution": best_flame_pos.tolist(),
            "fitness": float(best_flame_score),
            "current_fitness": current_best_val
        })

    return best_flame_pos.tolist(), float(best_flame_score), history

# -------------------------------
# Ruta principal
# -------------------------------
@views.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    
#try:
    
    num_machines = None
    data = request.form
    problem_type = data.get('problem_type', 'function')
    algo = data.get('algorithm')
    plot_data = None
    params = {}
    mode = data.get('optimization_type', 'min')  
    
    # --- PHASE 1: PREPARE THE PROBLEM ---
    if problem_type == 'TSP':
        raw_cities = data.get('tsp_cities', '').strip().split('\n')
        cities = []
        limit_lb = float(data.get('lb', 0))
        limit_ub = float(data.get('ub', 100))
        for line_no, line in enumerate(raw_cities, 1):
            line = line.strip()
            if not line: continue 
            
            try:
                parts = line.split(',')
                if len(parts) != 2:
                    raise ValueError(f"Formato incorrecto en línea {line_no}: debe ser x,y")
                
                x_c, y_c = map(float, parts)
                if not (limit_lb <= x_c <= limit_ub) or not (limit_lb <= y_c <= limit_ub):
                    return render_template("main.html", 
                        error=f"Ciudad en línea {line_no} ({x_c}, {y_c}) fuera de rango [{limit_lb}, {limit_ub}].")
                
                cities.append([x_c, y_c])
            except ValueError as e:
                return render_template("main.html", error=f"Error en datos de TSP: {str(e)}")
        if len(cities) < 3:
            return render_template("main.html", error="El TSP requiere al menos 3 ciudades para formar un ciclo.")
        
        dim = len(cities)
        lb, ub = 0, 1 
        expr_str = "Problema del Viajero"
        def objf(vec):
            route = np.argsort(vec)
            dist = 0
            for i in range(len(route)):
                c1 = cities[route[i]]
                c2 = cities[route[(i + 1) % len(route)]]
                dist += np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            
            # Si el modo es MAX, retornamos la distancia negativa 
            # (asumiendo que tus algoritmos siempre MINIMIZAN internamente)
            if mode == 'max':
                return -dist
            
            return dist
        
        plot_data = {"type": "tsp", "cities": cities}
        
    elif problem_type == 'KnS':
        raw_items = data.get('knapsack_items', '').strip().split('\n')
        capacity = float(data.get('knapsack_capacity', 15))
        items = []
        nombre_items = []
        
        for line in raw_items:
            if not line.strip(): continue
            parts = line.split(',')
            nombre_items.append(str(parts[0]))
            items.append([float(parts[1]), float(parts[2])])
        
        items = np.array(items)
        dim = len(items)
        lb, ub = 0, 1
        expr_str = "Problema de la Mochila"
        if mode == 'max':
            def objf(vec):
                selection = np.round(vec)
                total_value = np.sum(selection * items[:, 0])
                total_weight = np.sum(selection * items[:, 1])
                
                if total_weight > capacity:
                    exceso_discreto = total_weight - capacity
                    castigo_principal = 10000 * exceso_discreto
                    peso_continuo = np.sum(vec * items[:, 1])
                    return castigo_principal + (10 * peso_continuo)
                    
                return -total_value
        else: # Modo MIN: Buscar lo más barato, pero llenando la mochila
            def objf(vec):
                selection = np.round(vec)
                total_value = np.sum(selection * items[:, 0])
                total_weight = np.sum(selection * items[:, 1])
                
                # 1. Penalización por Exceso (Igual que antes)
                if total_weight > capacity:
                    exceso = total_weight - capacity
                    return 1e9 + (exceso * 1000) 
                # 2. Penalización por Mochila Vacía o poco llena
                # Calculamos cuánto espacio sobra
                espacio_libre = capacity - total_weight
                
                # El '1000' es el factor de peso. Debe ser lo suficientemente alto 
                # para que valga más la pena meter un objeto barato que dejar el hueco.
                return total_value + (espacio_libre * 1000)
        plot_data = {"type": "knapsack", "items_count": dim}
          
    
    elif problem_type == 'JobS':       
        raw_tasks = data.get('scheduling_tasks', '').strip().split('\n')
        num_machines = int(data.get('scheduling_machines', 3))
        
        duraciones = []
        nombres_tareas = []
        
        for line in raw_tasks:
            if not line.strip(): continue
            parts = line.split(',')
            nombres_tareas.append(parts[0].strip())
            duraciones.append(float(parts[1].strip()))
        
        dim = len(duraciones)
        lb, ub = 0, num_machines - 1
        
        def objf(vec):
            asignacion = np.round(vec).astype(int)
            cargas = np.zeros(num_machines)
            
            for i in range(len(asignacion)):
                m_idx = asignacion[i]
                m_idx = max(0, min(m_idx, num_machines - 1))
                cargas[m_idx] += duraciones[i]
            
            if mode == "min":
                # balancear: minimizar la máquina más cargada
                return float(np.max(cargas))
            elif mode == "max":
                # extender: maximizar la máquina más cargada
                return -float(np.max(cargas))  
    else:
        expr_str = data.get('function', 'x**2')
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace(' ', '')
        dim = int(data.get('dim', 1))
        lb = float(data.get('lb', -5))
        ub = float(data.get('ub', 5))
        x, y = symbols('x y')
        
        from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        if dim == 1:
            expr = parse_expr(expr_str, transformations=transformations).subs(y, 0)
            var_tuple = (x,)
        elif dim == 2:
            expr = parse_expr(expr_str, transformations=transformations)
            var_tuple = (x, y)
        else:
            expr = parse_expr(expr_str)
            var_tuple = symbols(f'x0:{dim}')
        f_sym = lambdify(var_tuple, expr, modules=["numpy", "math"])
        def f_original(vec):
            try:
                args = list(vec)
                res = f_sym(args[0]) if dim == 1 else f_sym(*args)
                
                final_val = float(res.evalf()) if hasattr(res, 'evalf') else float(res)
                
                if not np.isfinite(final_val):
                    return 1e18 
                return final_val
            except (ZeroDivisionError, OverflowError, TypeError):
                return 1e18 
        objf = (lambda v: -f_original(v)) if mode == "max" else f_original
        
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
    # --- DETERMINAR REPRESENTACIÓN BINARIA AUTOMÁTICA ---
    if problem_type in ['knapsack', 'bn_function']:
        repr_mode = "binary"
        lb, ub = 0, 1
    else:
        repr_mode = "continuous"
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
            "C1 (Cognitivo)": float(data.get('pso_c1', 1.5)),
            "C2 (Social)": float(data.get('pso_c2', 1.5)),
            "W (Inercia)": float(data.get('pso_w', 0.9))
        }
        best_sol, best_val, history = PSO(objf, lb, ub, dim, 
                                        params["Partículas"], params["Iteraciones"], 
                                        params["C1 (Cognitivo)"], params["C2 (Social)"], params["W (Inercia)"])
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
    elif algo == 'MFO':
        selection_mode = data.get('mfo_selection', 'Adaptive') 
        
        params = {
            "Polillas (N)": int(data.get('mfo_particles', 50)),
            "Iteraciones": int(data.get('mfo_iterations', 100)),
            "Constante Espiral": float(data.get('mfo_b', 1)),
            "Modo Selección": selection_mode
        }
        
        best_sol, best_val, history = MFO(
            objf, lb, ub, dim, 
            N=params["Polillas (N)"], 
            Max_iter=params["Iteraciones"],
            b=params["Constante Espiral"],
            selection_mode=selection_mode
        )
    else:
        return render_template("main.html", error="Algoritmo no soportado")
    # --- PHASE 3: FINAL TOUCHES ---
    if mode == "max":
        best_val = -best_val
        for h in history: 
            h["fitness"] = -h["fitness"]
            if "current_fitness" in h: 
                h["current_fitness"] = -h["current_fitness"]          
    
    datos_scheduling = []
    
    if problem_type == 'TSP':
        latex_func = None
    elif problem_type == 'KnS':
        latex_func = None
    elif problem_type == 'JobS':
        seleccion_final = np.round(best_sol)
        
        # 1. Create exactly 'num_machines' empty lists inside datos_scheduling
        datos_scheduling = [[] for _ in range(num_machines)]
        
        # 2. Safely assign tasks without nested loops
        for i in range(dim):
            m_idx = int(max(0, min(seleccion_final[i], num_machines - 1)))
            datos_scheduling[m_idx].append([nombres_tareas[i], duraciones[i]])  
            
        # 3. Calculate totals and insert them at index 0
        for machine in datos_scheduling:
            time_total = sum(task[1] for task in machine) # Pythonic way to sum
            machine.insert(0, time_total)
            
        latex_func = None
    else:
        latex_func = f"f(x) = {latex(expr)}" if dim == 1 else f"f(x, y) = {latex(expr)}"
    datos_mochila = [[], 0]
    if problem_type == 'KnS':
        seleccion_final = np.round(best_sol)
        peso_final = np.sum(seleccion_final * items[:, 1])
        
        solucion_plana = seleccion_final.flatten()
        for i in range(dim):
            datos_mochila[0].append({
                'nombre': nombre_items[i],
                'peso': float(items[i, 1]),
                'valor': float(items[i, 0]),
                'seleccion': bool(solucion_plana[i] >= 0.5) 
            }) 
        datos_mochila[1] = float(data.get('knapsack_capacity', 15))
    else:
        peso_final = None
    return render_template(
        'result.html.j2',
        algorithm=algo,
        problem=problem_type,
        best_solution=list(best_sol),
        best_value=float(best_val),
        history=history,
        latex_function=latex_func, 
        dim=dim,
        plot_data=plot_data,
        params=params,
        mode=mode,
        best_weight=peso_final,
        items_data=datos_mochila,
        scheduling_data=datos_scheduling,
        num_machines=num_machines,
    )
    #except Exception as e:
    return render_template("main.html.j2", error=f"Error Crítico: {str(e)}")