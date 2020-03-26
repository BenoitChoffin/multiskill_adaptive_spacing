import numpy as np
import os

def logistic(x):
    return 1/(1+np.exp(-x))

class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.

    From JJ's KTM repository: https://github.com/jilljenn/ktm.
    Modified to leave the possibility to get counters first at a t_1 and then at a t_2
    s.t. t_2 < t_1.

    Here, the order of the counters is correct. Shorter time windows (1/24, 1) are the first
    ones, whereas the larger time windows (7, 30) are at the end of get_counters output.

    Also, the time windows do not include their limit values (e.g., an attempt a day ago
    will not be counted as in time window 1)
    
    Finally, one must be careful **not to call** get_counters on a timestamp prior to one (or several)
    already pushed timestamps. What happens then is that get_counters considers that this attempt has
    just been done.
    """
    def __init__(self):
        self.queue = []
        self.window_lengths = [3600*24*30, 3600*24*7, 3600*24, 3600] # in seconds
        self.cursors = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        res = [len(self.queue)] + [len(self.queue) - cursor
                                    for cursor in self.cursors]
        return res[::-1]

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        self.cursors = [0] * len(self.window_lengths) # reset cursors, different from JJ's version
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1

def gen_outcome(alpha, delta, beta, win_counters, attempt_counters, h_features_win, h_features_att):
    """
   	Generate outcome from the DAS3H model with student ability alpha, item difficulty delta, (*single*)
   	skill difficulty beta, and attempt/win_counters as past study history and performance.
    
    Beware of the order of the counters because OurQueue will output the counters ranked by decreasing
    order of time window length.
    """
    return logistic(alpha+delta+beta+np.sum(np.log(1+np.array(attempt_counters))*np.array(h_features_att))+\
                    np.sum(np.log(1+np.array(win_counters))*np.array(h_features_win)))

def gen_cplx_outcome(alpha, delta, list_beta, list_win_counters, list_attempt_counters, list_h_features):
    """
    We talk about *complex outcome* here because the item can then involve multiple skills at the same
    time: hence the lists in the parameters of the function. The order of the skills in each of those
    lists must be the same. In each tuple of list_h_features, the odd thetas (win thetas) are assumed
    to be placed *before* the even thetas (attempt thetas).
    """
    res = alpha + delta
    for i in range(len(list_beta)):
        res += list_beta[i]
        res += np.sum(np.log(1+np.array(list_attempt_counters[i]))*np.array(list_h_features[i][1]))
        res += np.sum(np.log(1+np.array(list_win_counters[i]))*np.array(list_h_features[i][0]))
    return logistic(res)

def custom_argmax(arr):
    """
    Returns a random index from the set of argmax indices of an array (when there are ex aequo).
    """
    return np.random.choice(np.flatnonzero(arr == arr.max()))

def custom_argmin(arr):
    """
    Returns a random index from the set of argmin indices of an array (when there are ex aequo).
    """
    return np.random.choice(np.flatnonzero(arr == arr.min()))

def intersection(lst1, lst2):
    """
    Computes the intersection of two lists.

    Attention : when there are duplicates in the first list, they can be several times in the final result.
    """
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def prepare_folder(path):
    """Create folder from path."""
    if not os.path.isdir(path):
        os.makedirs(path)

def build_new_paths(FOLDER_NAME):
    """Create results folder path name."""
    RES_FOLDER = './results'
    XP_FOLDER = os.path.join(RES_FOLDER, FOLDER_NAME)
    return XP_FOLDER

def create_verbose_dico(review_strats_w_noreview, item_strategies):
    """Create dictionary to temporarily store experimental results."""
    verbose_dico = {}
    for item_strat in item_strategies:
        verbose_dico[item_strat]={}
        for strat_name in review_strats_w_noreview.keys():
            verbose_dico[item_strat][strat_name] = {}
            for param_value in review_strats_w_noreview[strat_name]:
                verbose_dico[item_strat][strat_name][param_value]={}
                for period in ["learning","retention"]:
                    verbose_dico[item_strat][strat_name][param_value][period] = []
    return verbose_dico

def create_distrib_attempts_dico(review_strats_w_random, item_strategies, B):
    distrib_attempts = {}
    for strat_name in review_strats_w_random.keys():
        distrib_attempts[strat_name]={}
        for param_value in review_strats_w_random[strat_name]:
            distrib_attempts[strat_name][param_value]={}
            for item_strat in item_strategies:
                distrib_attempts[strat_name][param_value][item_strat]={}
                for t in range(B):
                    distrib_attempts[strat_name][param_value][item_strat][t]={"att_skill_"+str(b):0 for b in range(B)}
    return distrib_attempts

def create_temp_recall_probas_dico(review_strats_w_noreview, item_strategies):
    temp_probas_dico = {}
    for strat_name in review_strats_w_noreview.keys():
        temp_probas_dico[strat_name]={}
        for param_value in review_strats_w_noreview[strat_name]:
            temp_probas_dico[strat_name][param_value]={}
            for item_strat in item_strategies:
                temp_probas_dico[strat_name][param_value][item_strat]={}
                for period in ["learning","retention"]:
                    temp_probas_dico[strat_name][param_value][item_strat][period] = []
    return temp_probas_dico

def print_run_perfs(verbose_dico):
    """
    Inside a run, print for each skill strat x param x item strat the ACPL and ACPR metrics.
    """
    for item_strat, v1 in verbose_dico.items():
        print(item_strat)
        list_of_perfs = []
        list_of_prints = []
        for skill_strat, v2 in v1.items():
            list_of_params = []
            acpl_perfs = []
            acpr_perfs = []
            for param_value, v3 in v2.items():
                list_of_params.append(param_value)
                for period, v4 in v3.items():
                    strat_perf = np.mean(verbose_dico[item_strat][skill_strat][param_value][period])
                    if period == "learning":
                        acpl_perfs.append(strat_perf)
                    if period == "retention":
                        acpr_perfs.append(strat_perf)
            # Find best parameters
            best_param_index = np.argmax(acpr_perfs)
            list_of_perfs.append(np.around(acpr_perfs[best_param_index],3))
            list_of_prints.append("\t {0:>26} | Best param : {1:>3} | ACPL = {2:>6} | ACPR = {3:>6}".format(skill_strat,
                np.around(list_of_params[best_param_index],2),
                np.around(acpl_perfs[best_param_index],3),
                np.around(acpr_perfs[best_param_index],3)))
        for strat_index in np.argsort(list_of_perfs)[::-1]:
            print(list_of_prints[strat_index])

def print_overall_perfs(results_df):
    print("\nOVERALL RESULTS \n")
    for item_strat in results_df["item_strategy"].unique():
        print(item_strat)
        list_of_perfs = []
        list_of_prints = []
        for skill_strat in results_df["skill_strategy"].unique():
            list_of_params = []
            acpl_perfs = []
            acpr_perfs = []
            for param_value in results_df[results_df["skill_strategy"]==skill_strat]["param"].unique():
                list_of_params.append(param_value)
                acpl_perfs.append(results_df[(results_df["skill_strategy"]==skill_strat) & \
                                             (results_df["param"]==param_value) & \
                                             (results_df["item_strategy"]==item_strat) & \
                                             (results_df["period"]=="learning")]["mean_recall"].mean())
                acpr_perfs.append(results_df[(results_df["skill_strategy"]==skill_strat) & \
                                             (results_df["param"]==param_value) & \
                                             (results_df["item_strategy"]==item_strat) & \
                                             (results_df["period"]=="retention")]["mean_recall"].mean())
            # Find best param
            best_param_index = np.argmax(acpr_perfs)
            list_of_perfs.append(np.around(acpr_perfs[best_param_index],3))
            list_of_prints.append("\t {0:>26} | Best param : {1:>3} | ACPL = {2:>6} | ACPR = {3:>6}".format(skill_strat,
                np.around(list_of_params[best_param_index],2),
                np.around(acpl_perfs[best_param_index],3),
                np.around(acpr_perfs[best_param_index],3)))
        for strat_index in np.argsort(list_of_perfs)[::-1]:
            print(list_of_prints[strat_index])

def store_results(main_dico, attempts_dico, temp_probas_dico, verbose_dico, review_strat, param, run_id, stud_id, t,
                  mean_recall, sum_attempts, recall_probs, relative_mean_recall, attempts_per_skill, B, item_strategy):
    """
    Function used to store experimental results inside several dictionaries.
    """
    main_dico["skill_strategy"].append(review_strat)
    main_dico["item_strategy"].append(item_strategy)
    main_dico["param"].append(param)
    main_dico["run_id"].append(run_id)
    main_dico["stud_id"].append(stud_id)
    main_dico["t"].append(t)
    main_dico["mean_recall"].append(np.around(mean_recall,5))
    main_dico["relative_mean_recall"].append(relative_mean_recall)
    main_dico["sum_attempts"].append(sum_attempts)
    for b in range(B):
        if (t < B) and (review_strat != "no_review"):
            attempts_dico[review_strat][param][item_strategy][t]["att_skill_"+str(b)] += attempts_per_skill[b]
    if t < B:
        temp_probas_dico[review_strat][param][item_strategy]["learning"].append(recall_probs)
        verbose_dico[item_strategy][review_strat][param]["learning"].append(mean_recall)
    else:
        temp_probas_dico[review_strat][param][item_strategy]["retention"].append(recall_probs)
        verbose_dico[item_strategy][review_strat][param]["retention"].append(mean_recall)
