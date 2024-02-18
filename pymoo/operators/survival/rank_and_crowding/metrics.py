import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
from pymoo.util.misc import find_duplicates
from pymoo.util.function_loader import load_function
from numpy import linalg as LA


def get_crowding_function(label):

    if label == "cd":
        fun = FunctionalDiversity(calc_crowding_distance, filter_out_duplicates=False)
    elif (label == "pcd") or (label == "pruning-cd"):
        fun = FunctionalDiversity(load_function("calc_pcd"), filter_out_duplicates=True)
    elif label == "bfe":
        fun = FunctionalDiversity(calc_balanceable_fitness_estimation, filter_out_duplicates=False)
    elif label == "ce":
        fun = FunctionalDiversity(calc_crowding_entropy, filter_out_duplicates=True)
    elif label == "mnn":
        fun = FuncionalDiversityMNN(load_function("calc_mnn"), filter_out_duplicates=True)
    elif label == "2nn":
        fun = FuncionalDiversityMNN(load_function("calc_2nn"), filter_out_duplicates=True)
    elif hasattr(label, "__call__"):
        fun = FunctionalDiversity(label, filter_out_duplicates=True)
    elif isinstance(label, CrowdingDiversity):
        fun = label
    else:
        raise KeyError("Crowding function not defined")
    return fun


class CrowdingDiversity:

    def do(self, F, n_remove=0):
        # Converting types Python int to Cython int would fail in some cases converting to long instead
        n_remove = np.intc(n_remove)
        F = np.array(F, dtype=np.double)
        return self._do(F, n_remove=n_remove)

    def _do(self, F, n_remove=None):
        pass


class FunctionalDiversity(CrowdingDiversity):

    def __init__(self, function=None, filter_out_duplicates=True):
        self.function = function
        self.filter_out_duplicates = filter_out_duplicates
        super().__init__()

    def _do(self, F, **kwargs):

        n_points, n_obj = F.shape

        if n_points <= 2:
            return np.full(n_points, np.inf)

        else:

            if self.filter_out_duplicates:
                # filter out solutions which are duplicates - duplicates get a zero finally
                is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
            else:
                # set every point to be unique without checking it
                is_unique = np.arange(n_points)

            # index the unique points of the array
            _F = F[is_unique]

            _d = self.function(_F, **kwargs)

            d = np.zeros(n_points)
            d[is_unique] = _d

        return d


class FuncionalDiversityMNN(FunctionalDiversity):

    def _do(self, F, **kwargs):

        n_points, n_obj = F.shape

        if n_points <= n_obj:
            return np.full(n_points, np.inf)

        else:
            return super()._do(F, **kwargs)


def calc_crowding_distance(F, **kwargs):
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return cd


def calc_balanceable_fitness_estimation(F, **kwargs):
    """
      BFE (Balanceable Fitness Estimation)
      return the values of BFE for each paricle in F
    """

    def Euclidean_distance(point_1, point_2):
        """
        calculate the Euclidean distance between two points

        Parameters:
        - point1: list like [x1, x2, x3, ... ]
        - point2: list like [x1, x2, x3, ... ]

        Returns:
        - The Euclidean distance between the two points.
        """
        if len(point_1) != len(point_2):
            return 'there is diff in dimantion !!!'
        else:
            sum_ = 0
            for i in range(len(point_1)):
                x1 = point_1[i]
                x2 = point_2[i]
                sum_ += (x2 - x1) ** 2
            return math.sqrt(sum_)

    def normalize(value, min_, max_):
        return (value - min_) / (max_ - min_)

    def apply_normalize_on_objs(pop_objectives, ideal_point):
        """
        input
        """
        list_0 = []
        for pop in pop_objectives:
            list_1 = [Euclidean_distance([x1], [x2]) for x1, x2 in zip(pop, ideal_point)]
            list_0.append(list_1)

        # Transpose the list to work with columns
        transposed_example = list(zip(*list_0))

        # Find the max and min for each column
        max_values = [max(column) for column in transposed_example]
        min_values = [min(column) for column in transposed_example]
        normalized = []
        for obj in list_0:
            normalize_ = [normalize(value, min_, max_) for value, min_, max_ in zip(obj, min_values, max_values)]
            normalized.append(normalize_)
        return normalized

    def calculate_convergence(pop):
        '''
        pop: list where each item contains list of objective functions
        '''

        def dis(list_objectives):
            '''
            input:
            list_objective: normalized objectives
            '''
            objectives_p_2 = [item ** 2 for item in list_objectives]
            return math.sqrt(sum(objectives_p_2))

        # convergence ---
        convergence_list = []
        for p_i in pop:
            m = len(p_i)
            c_v = 1 - (dis(p_i) / math.sqrt(m))
            convergence_list.append(c_v)
        return convergence_list

    def calculate_SDE(normalized_objs):
        """
        caluculate the Shift-Based Density
        """
        SDE_s = []

        def sde_0(value1, value2):
            if value2 > value1:
                return value2 - value1
            else:
                return 0

        for p_i in normalized_objs:
            collect_sde = []
            list_ = normalized_objs.copy()
            list_.remove(p_i)
            for p_j in list_:
                collect_sde.append(math.sqrt(sum([sde_0(i, j) ** 2 for i, j in zip(p_i, p_j)])))

            SDE_i = min(collect_sde)
            SDE_s.append(SDE_i)
        return SDE_s

    def calculate_diversity(normalized_objs):
        '''
        pop: list where each item contains list of objective functions
        '''
        SDE_s = calculate_SDE(normalized_objs)
        SDE_max = max(SDE_s)
        SDE_min = min(SDE_s)
        cd_s = []
        for SDE_i in SDE_s:
            cd_i = (SDE_i - SDE_min) / (SDE_max - SDE_min)
            cd_s.append(cd_i)
        return cd_s

    def calculate_d1_d2(objectives_pop, z_star, z_end):
        """
        calculate d1
        """
        d1_s = []
        d2_s = []
        z_star = np.array(z_star)
        z_end = np.array(z_end)
        for point in objectives_pop:
            d1 = (np.inner((np.array(point) - z_star), (z_end - z_star))) / LA.norm(z_end - z_star)
            d1_s.append(d1)
            try:

                d2 = math.sqrt(np.inner((np.array(point) - z_star), (np.array(point) - z_star)) - d1**2)
            except:
                print(f'the point = {point} , z_star: {z_star}, d1: {d1}')
                print(f'point - z_star {(np.array(point) - z_star)}')
                print(f'the inner for two array {np.inner((np.array(point) - z_star), (np.array(point) - z_star))}')

                d2= 0
            d2_s.append(d2)
        return d1_s, d2_s

    dic_boundaries = {
        'mean_cd': 0,
        'mean_cv': 0,
        'mean_d1': 0,
        'mean_d2': 0,
    }

    def get_boundaries(cv_values, cd_values, d1_values, d2_values):
        """
        return the boundaries (mean_cv, mean_cd, mean_d1, mean_d2)
        """
        dic_boundaries['mean_cv'] = sum(cv_values) / len(cv_values)
        dic_boundaries['mean_cd'] = sum(cd_values) / len(cd_values)
        dic_boundaries['mean_d1'] = sum(d1_values) / len(d1_values)
        dic_boundaries['mean_d2'] = sum(d2_values) / len(d2_values)

        return dic_boundaries

    import random

    cases = {
        '1.1.1': {'alpha': random.uniform(0.6, 1.3), 'beta': 1},
        '1.1.2': {'alpha': 1, 'beta': 1},
        '1.2.1': {'alpha': 0.6, 'beta': 0.9},
        '1.2.2': {'alpha': 0.9, 'beta': 0.9},
        '2.1.1': {'alpha': random.uniform(0.6, 1.3), 'beta': random.uniform(0.6, 1.3)},
        '2.1.2': {'alpha': 1, 'beta': 1},
        '2.2': {'alpha': 0.2, 'beta': 0.2},
    }

    def get_alpha_beta(cv, cd, d1, d2, boundaries):
        """
        Return the values of alpha and beta based on cv and cd

        input:
        - cv: value of particle's convergance
        - cd: value of particle's diversity
        - boundaries: dictionary contains (mean_cv, mean_cd, mean_d1, mean_d2)

        ouput:
        - alpha , beta
        """
        # case 1
        if cv < boundaries['mean_cv']:
            # case 1.1
            if d1 < boundaries['mean_d1']:
                # case 1.1.1
                if cd < boundaries['mean_cd']:
                    return cases['1.1.1']
                # case 1.1.2
                else:
                    return cases['1.1.2']

            # case 1.2
            else:
                # case 1.2.1
                if cd < boundaries['mean_cd']:
                    return cases['1.2.1']
                # case 1.2.2
                else:
                    return cases['1.2.2']

        # case 2
        else:
            # case 2.1
            if d1 < boundaries['mean_d1'] and d2 >= boundaries['mean_d2']:
                # case 2.1.1
                if cd < boundaries['mean_cd']:
                    return cases['2.1.1']
                # case 2.1.2
                else:
                    return cases['2.1.2']

            # case 2.2
            elif d1 >= boundaries['mean_d1'] or d2 < boundaries['mean_d2']:
                return cases['2.2']

    # ideal_point = [-0.26386886936020537, -0.038662102103408015, -12685.936307382268, -7.788507340173738]
    # z_end = [5.873340084744861, 0.17553681652439818, 4516.724857386221, 0.898005768872491]

    ideal_point = [0, 0]
    z_end = [1, 1]

    BFE = []
    normalized_objectives = apply_normalize_on_objs(F, ideal_point)
    cd_s = calculate_convergence(normalized_objectives)
    cv_s = calculate_diversity(normalized_objectives)
    d1_s, d2_s = calculate_d1_d2(F, ideal_point, z_end)
    boundaries = get_boundaries(cv_values=cv_s,
                                cd_values=cd_s,
                                d1_values=d1_s,
                                d2_values=d2_s)

    for index, particle in enumerate(F):
        # calculate value of diversity and convergence for this particle
        c_d = cd_s[index]
        c_v = cv_s[index]
        d1 = d1_s[index]
        d2 = d2_s[index]
        # calculate alpha and beta for this particle
        a_b_dic = get_alpha_beta(c_d, c_v, d1, d2, boundaries)

        _BFE = a_b_dic['alpha'] * c_d + a_b_dic['beta'] * c_v
        BFE.append(_BFE)
    return BFE

def calc_crowding_entropy(F, **kwargs):
    """Wang, Y.-N., Wu, L.-H. & Yuan, X.-F., 2010. Multi-objective self-adaptive differential 
    evolution with elitist archive and crowding entropy-based diversity measure. 
    Soft Comput., 14(3), pp. 193-209.

    Parameters
    ----------
    F : 2d array like
        Objective functions.

    Returns
    -------
    ce : 1d array
        Crowding Entropies
    """
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dl = dist.copy()[:-1]
    du = dist.copy()[1:]

    # Fix nan
    dl[np.isnan(dl)] = 0.0
    du[np.isnan(du)] = 0.0

    # Total distance
    cd = dl + du

    # Get relative positions
    pl = (dl[1:-1] / cd[1:-1])
    pu = (du[1:-1] / cd[1:-1])

    # Entropy
    entropy = np.row_stack([np.full(n_obj, np.inf),
                            -(pl * np.log2(pl) + pu * np.log2(pu)),
                            np.full(n_obj, np.inf)])

    # Crowding entropy
    J = np.argsort(I, axis=0)
    _cej = cd[J, np.arange(n_obj)] * entropy[J, np.arange(n_obj)] / norm
    _cej[np.isnan(_cej)] = 0.0
    ce = _cej.sum(axis=1)

    return ce


def calc_mnn_fast(F, **kwargs):
    return _calc_mnn_fast(F, F.shape[1], **kwargs)


def calc_2nn_fast(F, **kwargs):
    return _calc_mnn_fast(F, 2, **kwargs)


def _calc_mnn_fast(F, n_neighbors, **kwargs):

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = 1.0

    # F normalized
    F = (F - F.min(axis=0)) / norm

    # Distances pairwise (Inefficient)
    D = squareform(pdist(F, metric="sqeuclidean"))

    # M neighbors
    M = F.shape[1]
    _D = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]

    # Metric d
    d = np.prod(_D, axis=1)

    # Set top performers as np.inf
    _extremes = np.concatenate((np.argmin(F, axis=0), np.argmax(F, axis=0)))
    d[_extremes] = np.inf

    return d
