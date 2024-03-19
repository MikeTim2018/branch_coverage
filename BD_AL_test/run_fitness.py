from lib2to3.pytree import convert
from pyswarms.single import GlobalBestPSO as GBPSO
from pyswarms.single import LocalBestPSO as LBPSO
import time
import random
import ast
import numpy as np
import pandas as pd
from tree_converter import TreeVisitor
from test_fitness import Fitness
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
import asyncio
from concurrent.futures import ProcessPoolExecutor


paths = {
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/rock_paper_scissor_player_choice.py": [1],
    "/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/bounce_draw.py": [2, -300, 300],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/guess_the_number_input_guess.py": [1, 0, 6],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/jogo_da_velha_python_actualizar_jogadas.py": [1, 0, 10],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/rock_paper_scissor_number_to_name.py": [1, 0, 5],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/test_game_programs/function_only_testings/TRPG_character_create_character.py": [1, 0, 4],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/bubble_sort.py": [4],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/minimum.py": [4],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/three_number_sort.py": [3],
    #"/Volumes/develop/Msc.Computo_aplicado/branch_coverage/BD_AL_test/test_programs/trig_area.py": [3, 0, 2]
}

algorithms = [PSO(pop_size=100, w=0.5, c1 = 0.3, c2 = 0.9),
              GA(pop_size=100, eliminate_duplicates=True),
              DE(
                 pop_size=100,
                 sampling=LHS(),
                 variant="DE/rand/1/bin",
                 CR=0.3,
                 dither="vector",
                 jitter=False
                ),
              NelderMead(),
              PatternSearch(),
              SRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2),
              ISRES(n_offsprings=200, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)]
# algorithm = CMAES(x0=np.random.random(dimensions))  # Only for more than 1 dimension



def run_algorithm(algorithm, problem=None):
    """
    Function that runs the selected algorithm into the tree nodes
    """
    random_number = random.randint(0, 99999999)
    try:
        cost, pos = algorithm.optimize(fitness.fitness_function, iters=100)
        return cost, pos
    except:
        result = minimize(
                          problem, 
                          algorithm, 
                          ('n_gen', 100),
                          seed=random_number,                         
                          verbose=True)
    return result.F, result.X

def convert_tree(path):
    """
    Function to convert the selected function file into tree traversable nodes
    """
    with open(path, 'r+') as filename:
       lines = filename.readlines()
       tree = ast.parse(''.join(lines))
    # print(ast.dump(tree))
    tree = ast.parse(tree)
    visitor = TreeVisitor()
    visitor.visit(tree)
    print(visitor.nodes)
    return visitor

class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_cost = Column("x_mean", width=13)
        self.x_std = Column("x_std", width=13)
        self.columns += [self.x_mean, self.x_std]

    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        self.x_std.set(np.std(algorithm.pop.get("X")))

class FitnessProblem(Problem):

    def __init__(self, fitness=None, dimensions=None, xl=-99999, xu=99999):
        self.fitness = fitness
        self.ndim = dimensions
        super().__init__(n_var=dimensions, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fitness.fitness_function(x)


async def run_async_fitness(visitor, dimensions, algorithm, path):
    results_df = pd.DataFrame()
    algorithms_list = []
    programs = []
    coverages = []
    times = []
    iteration = []
    for i in range(1, 31):
        max_paths = 0
        coverage = 0
        past_walking = []
        best_positions = {}
        more_paths = True
        fitness = Fitness(visitor)
        xl =  dimensions[1] if len(dimensions) > 1 else None
        xu = dimensions[2] if len(dimensions) > 2 else None
        if xu is not None and xl is not None:
            problem = FitnessProblem(fitness, dimensions[0], xl=xl, xu=xu)
        elif xu and not xl:
            problem = FitnessProblem(fitness, dimensions[0], xu=xu)
        elif xl and not xu:
            problem = FitnessProblem(fitness, dimensions[0], xl=xl)
        else:
            problem = FitnessProblem(fitness, dimensions[0])
        iteration.append(i)
        st = time.time()
        programs.append(path.split('/')[-1])
        algorithms_list.append(algorithm.__module__)
        # algorithm = algorithms[1]
        # algorithm = CMAES(x0=np.random.random(dimensions))  # Only for more than 1 dimension
        # gbpso = GBPSO(particles,dimensions,options=options)
        while more_paths:
            #cost, pos = run_algorithm(gbpso)
            cost, pos = run_algorithm(algorithm, problem)
            particle_pos = np.array([pos],np.float32)
            fitness.resolve_path(particle_pos)
            has_path = lambda x: lambda y: y in x
            if all(map(has_path(list(set(past_walking))),fitness.current_walked_tree)) and past_walking:
                break
            if not fitness.current_walked_tree:
                break
            coverage = len(list(set(fitness.walked_tree)))/len(fitness.whole_tree)
            best_positions.update({f"{pos}": f"Cost is {cost} and coverage is {coverage}"})
            print(f"Real coverage is {coverage}")
            past_walking.extend(fitness.current_walked_tree)
            # plot_cost_history(cost_history=gbpso.cost_history)
            # plt.show()
        print(fitness.custom_weights)
        print(f"Positions and coverage are {best_positions}")
        print(f"The coverage of the matrix is {coverage*100}%")
        print(f"whole tree is {list(set(fitness.whole_tree))} Walked tree:  {list(set(fitness.walked_tree))}")
        et = time.time()
        total_time = et - st
        times.append(total_time)
        coverages.append(f"{coverage*100}%")
        print(f"Total elapsed time is {total_time} seconds")
    results_df['Algorithm'] = algorithms_list
    results_df['Iteration'] = iteration
    results_df["Code Tested"] = programs
    results_df['Coverage'] = coverages
    results_df['Execution Time'] = times
    results_df.to_csv(f"output_results_{path.split('/')[-1]}_{algorithm}.csv", index=False)
    return results_df


async def query_concurrently(visitor, dimensions, path):
    """ Start concurrent tasks by start and end sequence number """
    tasks = []
    for algorithm in algorithms:
        tasks.append(asyncio.create_task(run_async_fitness(visitor, dimensions, algorithm, path)))
    results = await asyncio.gather(*tasks)
    return results

def run_batch_tasks(visitor, dimensions, path):
    """ Execute batch tasks in sub processes """

    results = [result for result in asyncio.run(query_concurrently(visitor, dimensions, path))]
    return results

async def main():
    """ Distribute tasks in batches to be executed in sub-processes """
    start = time.monotonic()

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        for path, dimensions in paths.items():
            #path = "test_programs/test_game_programs/function_only_testings/bounce_draw.py"
            visitor = convert_tree(path)
            tasks = loop.run_in_executor(executor, run_batch_tasks, visitor, dimensions, path)                 

    results = [result for sub_list in await asyncio.gather(*tasks) for result in sub_list]

    print(f"We get {len(results)} results. All last {time.monotonic() - start:.2f} second(s)")

if __name__ == '__main__':
    # 30 executions per algorithm
    results_df = pd.DataFrame()
    algorithms_list = []
    programs = []
    coverages = []
    times = []
    iteration = []
    asyncio.run(main())  