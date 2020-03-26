from copy import deepcopy
import numpy as np
from utils import gen_outcome, intersection

def greedy_skill_selection(week, B, list_of_ri, stud_alpha, avg_deltas, skill_betas, win_params, att_params, q):
    """
    Select skill with largest conditional marginal learning gain (cf. Hunziker et al., 2018).
    Return selected skill id.

    Concretely, we compare the impact on future retention performance of selecting any of the available
    skills. We choose the skill that will bring the highest expected utility during retention. Since skills are
    independent, we can just compare the effect of one attempt of each skill on the future correctness probability
    of *that* skill.
    """
    cumulative_gains = []
    for b in range(week): # Loops over all the skills that are available to review
        probas_recall_no_review = [] # No review
        probas_recall_review_1 = [] # Review + outcome = 1
        probas_recall_review_0 = [] # Review + outcome = 0
        # Copy queues
        q_no_review = deepcopy(q)
        q_review_1 = deepcopy(q)
        q_review_0 = deepcopy(q)
        # Simulate attempt/win on the queues
        q_review_1[(b,"attempts")].push(week*3600*24*7) # Simulate attempt on *that* skill on that week
        q_review_0[(b,"attempts")].push(week*3600*24*7) # Simulate attempt on *that* skill on that week
        q_review_1[(b,"wins")].push(week*3600*24*7) # Simulate win on *that* skill on that week
        # Generate outcome for expectation on y_t
        win_counters = q_no_review[(b,"wins")].get_counters(week*3600*24*7)
        attempt_counters = q_no_review[(b,"attempts")].get_counters(week*3600*24*7)
        proba_outcome_1 = gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters,
                                      attempt_counters, win_params[b], att_params[b])
        for t in B-1+np.array(list_of_ri): # we only count for the retention period
            # Attempt counters
            attempt_counters_no_review = q_no_review[(b,"attempts")].get_counters(t*3600*24*7)
            attempt_counters_review_1 = q_review_1[(b,"attempts")].get_counters(t*3600*24*7)
            attempt_counters_review_0 = q_review_0[(b,"attempts")].get_counters(t*3600*24*7)
            # Win counters
            win_counters_no_review = q_no_review[(b,"wins")].get_counters(t*3600*24*7)
            win_counters_review_1 = q_review_1[(b,"wins")].get_counters(t*3600*24*7)
            win_counters_review_0 = q_review_0[(b,"wins")].get_counters(t*3600*24*7)
            # Proba of recall
            probas_recall_no_review.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_no_review,
                                                       attempt_counters_no_review, win_params[b], att_params[b]))
            probas_recall_review_1.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_review_1,
                                                      attempt_counters_review_1, win_params[b], att_params[b]))
            probas_recall_review_0.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_review_0,
                                                      attempt_counters_review_0, win_params[b], att_params[b]))
        cumulative_gains.append(proba_outcome_1*np.mean(probas_recall_review_1)+ \
                                (1-proba_outcome_1)*np.mean(probas_recall_review_0)-np.mean(probas_recall_no_review))
    return np.argmax(cumulative_gains)

def greedy_multiskill_selection(week, B, list_of_ri, stud_alpha, avg_deltas, skill_betas, win_params, att_params, q,
                                q_mat, inv_q_mat, train_items_per_skill):
    """
    Select *combination of skills* with largest conditional marginal learning gain (cf. Hunziker et al., 2018).
    Return selected item id.

    Multiskill version of the greedy_skill_selection function above.
    """
    cumulative_gains = []
    for b in range(week): # Loops over all the skills that are available to review
        probas_recall_no_review = [] # No review
        probas_recall_review_1 = [] # Review + outcome = 1
        probas_recall_review_0 = [] # Review + outcome = 0
        # Copy queues
        q_no_review = deepcopy(q)
        q_review_1 = deepcopy(q)
        q_review_0 = deepcopy(q)
        # Simulate attempt/win on the queues
        q_review_1[(b,"attempts")].push(week*3600*24*7) # Simulate attempt on *that* skill on that week
        q_review_0[(b,"attempts")].push(week*3600*24*7) # Simulate attempt on *that* skill on that week
        q_review_1[(b,"wins")].push(week*3600*24*7) # Simulate win on *that* skill on that week
        # Generate outcome for expectation on y_t
        win_counters = q_no_review[(b,"wins")].get_counters(week*3600*24*7)
        attempt_counters = q_no_review[(b,"attempts")].get_counters(week*3600*24*7)
        proba_outcome_1 = gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters,
                                      attempt_counters, win_params[b], att_params[b])
        for t in B-1+np.array(list_of_ri): # we only count for the retention period
            # Attempt counters
            attempt_counters_no_review = q_no_review[(b,"attempts")].get_counters(t*3600*24*7)
            attempt_counters_review_1 = q_review_1[(b,"attempts")].get_counters(t*3600*24*7)
            attempt_counters_review_0 = q_review_0[(b,"attempts")].get_counters(t*3600*24*7)
            # Win counters
            win_counters_no_review = q_no_review[(b,"wins")].get_counters(t*3600*24*7)
            win_counters_review_1 = q_review_1[(b,"wins")].get_counters(t*3600*24*7)
            win_counters_review_0 = q_review_0[(b,"wins")].get_counters(t*3600*24*7)
            # Proba of recall
            probas_recall_no_review.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_no_review,
                                                       attempt_counters_no_review, win_params[b], att_params[b]))
            probas_recall_review_1.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_review_1,
                                                      attempt_counters_review_1, win_params[b], att_params[b]))
            probas_recall_review_0.append(gen_outcome(stud_alpha, avg_deltas, skill_betas[b], win_counters_review_0,
                                                      attempt_counters_review_0, win_params[b], att_params[b]))
        cumulative_gains.append(proba_outcome_1*np.mean(probas_recall_review_1)+ \
                                (1-proba_outcome_1)*np.mean(probas_recall_review_0)-np.mean(probas_recall_no_review))
    
    selected_skills = []
    selected_skills.append(np.argmax(cumulative_gains))
    # Then, selection of the other skills
    # We need to loop over all different possible skills, in the decreasing order of expected gain
    decreasing_order = np.argsort(cumulative_gains)[::-1]
    acceptable_items = [item_id for item_id in inv_q_mat[selected_skills[0]] if item_id < week*train_items_per_skill]
    for k in range(1,len(cumulative_gains)):
        intersec_items = intersection(acceptable_items,inv_q_mat[decreasing_order[k]])
        if len(intersec_items) == 0:
            continue
        else:
            acceptable_items = intersec_items.copy()
    # Choose item
    # There is at least 1 item in acceptable_items
    selected_item = np.random.choice(acceptable_items)
    return selected_item
