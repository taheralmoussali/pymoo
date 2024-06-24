import numpy as np
import warnings
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from scipy.optimize import dual_annealing
import numpy as np

def simulated_annealing(func, x0, bounds, max_iter=100, initial_temp=100, cooling_rate=0.99, min_temp=1e-3):
    # Initialize variables
    current_solution = x0
    current_value = func(current_solution)
    best_solution = current_solution
    best_value = current_value
    temp = initial_temp

    # Convert bounds to a NumPy array if it's a list of tuples
    bounds = np.array(bounds)

    # Dominator instance
    dominator = Dominator()

    # Annealing process
    for i in range(max_iter):
        # Generate a new candidate solution
        candidate_solution = current_solution + np.random.uniform(-0.1, 0.1, size=len(x0))
        
        # Ensure the candidate solution is within bounds
        candidate_solution = np.clip(candidate_solution, bounds[:, 0], bounds[:, 1])
        
        candidate_value = func(candidate_solution)

        # Decide whether to accept the candidate solution based on dominance
        if dominator.get_relation(candidate_value, current_value) == -1 or np.random.rand() < np.exp((np.sum(current_value) - np.sum(candidate_value)) / temp):
            current_solution = candidate_solution
            current_value = candidate_value

        # Update the best solution found based on dominance
        if dominator.get_relation(candidate_value, best_value) == -1:
            best_solution = candidate_solution
            best_value = candidate_value

        # Decrease the temperature
        temp *= cooling_rate

        # If the temperature is too low, break the loop
        if temp < min_temp:
            break

    return best_solution


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------

def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

class RankAndCrowdingSurvival(RankAndCrowding):
    
    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
                "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead, which supports several and custom crowding diversity metrics.",
                DeprecationWarning, 2
            )
        super().__init__(nds, crowding_func)

# =========================================================================================================
# Implementation
# =========================================================================================================

class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

class NSGA2_SA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)

        # Apply Simulated Annealing to each solution in the population
        for ind in self.pop:
            x0 = ind.X  # Current solution
            bounds = [(lb, ub) for lb, ub in zip(self.problem.xl, self.problem.xu)]  # Variable bounds
            func = lambda x: self.problem.evaluate(x)  # Objective function

            new_x = simulated_annealing(func, x0, bounds)  # Apply standard SA
            ind.set("X", new_x)  # Update solution
            self.evaluator.eval(self.problem, ind)  # Re-evaluate solution

parse_doc_string(NSGA2.__init__)
