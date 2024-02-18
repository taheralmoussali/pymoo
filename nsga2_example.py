import sys
# sys.path.append('F:\\Rachis_systems\\14- hyperledger\\sharding\\pymoo')

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
num_obj = 2
ref_dirs = get_reference_directions("das-dennis", num_obj, n_partitions=50)

problem = get_problem('zdt1')


# algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs,survival=)
algorithm = NSGA2(pop_size=100, )

res = minimize(problem= problem,
               algorithm= algorithm,
               seed = 1,
               verbose = False)
plot = Scatter()
plot.add(problem.pareto_front(), plot_type='line', color='black', alpha=0.7)
plot.add(res.F, facecolor='none', edgecolor='red')
plot.show()



#
# import numpy as np
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.problems import get_problem
# from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
# from pymoo.core.callback import Callback
#
# class MyCallback(Callback):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data = []
#
#     def notify(self, algorithm):
#         # Store the current population for plotting
#         self.data.append(algorithm.pop.get("X"))
#
# # Define ZDT1 problem
# problem = get_problem("zdt1")
#
# # Define the algorithm (NSGA-II) and attach the callback
# algorithm = NSGA2(
#     pop_size=100,
#     crossover_probability=0.9,
#     mutation_probability=1.0 / problem.n_var,
#     eliminate_duplicates=True
# )
#
# # Define the termination criteria
# termination = ("n_gen", 200)
#
# # Initialize the callback
# callback = MyCallback()
#
# # Run the optimization with the callback
# results = minimize(problem, algorithm, termination=termination, callback=callback, seed=1, verbose=True)
# print(callback.data.keys())
# # Visualize the population for each iteration
# # for i, population in enumerate(callback.data):
# #     plot = Scatter(title=f"ZDT1 - NSGA-II - Iteration {i + 1}")
# #     plot.add(population, color="blue", alpha=0.5, s=30, label="Population")
# #     plot.show()
#
