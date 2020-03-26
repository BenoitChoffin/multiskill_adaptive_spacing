import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from utils import gen_cplx_outcome, OurQueue
import matplotlib.pyplot as plt

def generate_q_matrix(B, items_per_skill, max_add_skills=2):
    """
    Randomly generate q-matrix for item-skill relationships. Only one type of item is considered:
    train items during practice (performance is measured by means of the raw correctness probas).
    
    Training items are used during the reviewing phase. We simulate initial learning of a block by
    adding a single attempt to this block for the student. Then, for review purposes, the student uses
    the train items. The train items q-matrix is block-diagonal: for each block b, we forcefully tag
    items_per_skill items with that skill b; then a random number of additional tags is generated
    and tags among the b-1 previous blocks can then be added to the q-matrix. This simulates a real-world
    curriculum in which more complex exercises build upon the previously encountered learning blocks AND
    on the current block b (prerequisite relationships).
    
    Param max_add_skills determines the nb of additional train skill tags for a given item in the
    q-matrix.
    
    Output: 2 dictionaries (q-mat, inv q-mat). Keys from the q-matrices are the
    item indices and values are lists containing the skill indices that they train. This is
    the inverse for the inv q-mat.
    """
    qmat = {} ; inv_qmat = {b:set() for b in range(B)}
    for i in range(B):
        for j in range(items_per_skill):
            qmat[i*items_per_skill+j] = [i]
            # We can have maximum max_add_skills+1 skill tags for a given exercise, provided that the number of
            # alread seen skills (minus 1) is higher than max_add_skills+1
            nb_of_additional_tags = np.random.randint(low=0, high=min(i+1,max_add_skills+1))
            # Pick nb_of_additional_tags skills among the available skill tags
            add_skill_tags = np.random.choice(i,nb_of_additional_tags,replace=False)
            for l in add_skill_tags:
                qmat[i*items_per_skill+j].append(l)
    # Build reverse q-matrix
    for item_id, list_of_skill_ids in qmat.items():
        for skill_id in list_of_skill_ids:
            inv_qmat[skill_id].add(item_id)
    inv_qmat = {k:list(v) for k,v in inv_qmat.items()}
    return qmat, inv_qmat

def gen_random_das3h_curve(nb_of_curves, zone="1"):
    """
    Output 2*nb_of_curves sets of thetas parameters (odd and even thetas), sampled from the specified
    sampling zone (cf. IJAIED paper).

    Odd thetas = win thetas; even thetas = attempt thetas.
    """
    if zone == "6": # Win thetas must be negative and smaller in absolute value than att thetas
        even_thetas = np.random.uniform(0,2,(nb_of_curves,5))
        odd_thetas = np.random.uniform(-even_thetas,0,(nb_of_curves,5))
    else: # by default, zone 1 ; both parameters must be positive
        even_thetas = np.random.uniform(0,2,(nb_of_curves,5))
        odd_thetas = np.random.uniform(0,2,(nb_of_curves,5))
    return odd_thetas, even_thetas

def estimate_das3h_params(true_item_deltas, true_skill_betas, true_odd_thetas, true_even_thetas, qmat,
                          nb_of_students, learning_sessions, interactions_per_session,
                          deterministic_outcome=True):
    """
    Simulate students answering items and estimate parameters. Student parameters are generated according
    to the same distribution as in the main experiment. Students do not solve exercises in the same order
    as in the main experiment: we assume that they all have prior knowledge of the skills they review.

    Input:
        * true_item_deltas, true_skill_betas, true_odd_thetas, true_even_thetas: true model parameters,
        like in experiment.das3h_multiskill_simul;
        * qmat: q-matrix, as gen_random_das3h_curve outputs;
        * nb_of_students: number of simulated students for *this estimation step*;
        * learning_sessions: number of learning sessions underwent by each of the simulated students of
        *this estimation step*. Each learning session consists in multiple interactions that are temporally
        close;
        * interactions_per_session: number of interactions that each student will have in each learning session;
        * deterministic_outcome: if True, each student behaves deterministically (if P(correct)>.5, then correct);

    Output: estimated delta, beta, and theta parameters.
    """
    res = {"stud_id":[], "item_id":[], "attempts":[], "wins":[], "outcome":[]}
    # Build sparse Q-matrix
    sparse_qmat = np.zeros((len(true_item_deltas),len(true_skill_betas)))
    for k,v in qmat.items():
        for kc in v:
            sparse_qmat[k,kc] = 1
    for i in range(nb_of_students):
        stud_alpha = np.random.normal(loc=0,scale=1,size=1)[0]
        t = 0 # counted in seconds
        q = defaultdict(lambda: OurQueue())
        for j in range(learning_sessions):
            for l in range(interactions_per_session):
                t += np.random.randint(60)*60 # minutes converted to seconds
                item_id = np.random.randint(len(true_item_deltas))
                temp_skills = []
                temp_betas = []
                temp_win_counters = []
                temp_attempt_counters = []
                temp_h_features = []
                win_counters_to_save = np.zeros(len(true_skill_betas)*5)
                attempt_counters_to_save = np.zeros(len(true_skill_betas)*5)
                for b in qmat[item_id]:
                    temp_skills.append(b)
                    temp_betas.append(true_skill_betas[b])
                    temp_h_features.append((true_odd_thetas[b],true_even_thetas[b]))
                    temp_attempt_counters.append(list(np.log(1+np.array(q[(b,"attempts")].get_counters(t)))))
                    attempt_counters_to_save[b*5:(b+1)*5] = np.array(temp_attempt_counters[-1])
                    temp_win_counters.append(list(np.log(1+np.array(q[(b,"wins")].get_counters(t)))))
                    win_counters_to_save[b*5:(b+1)*5] = np.array(temp_win_counters[-1])
                    q[(b,"attempts")].push(t)
                outcome = gen_cplx_outcome(stud_alpha, true_item_deltas[item_id], temp_betas,
                                           temp_win_counters, temp_attempt_counters, temp_h_features)
                if deterministic_outcome:
                    final_outcome = np.around(outcome)
                    if outcome > .5:
                        for b in temp_skills:
                            q[(b,"wins")].push(t)
                else:
                    final_outcome = np.random.choice(2,p=[1-outcome,outcome])
                    if final_outcome:
                        for b in temp_skills:
                            q[(b,"wins")].push(t)
                res["stud_id"].append(i)
                res["item_id"].append(item_id)
                res["wins"].append(list(win_counters_to_save))
                res["attempts"].append(list(attempt_counters_to_save))
                res["outcome"].append(final_outcome)
            t += 7*24*3600
    # Form training dataset
    onehot = OneHotEncoder(categories="auto")
    sparse_df = onehot.fit_transform(np.array(res["stud_id"]).reshape(-1,1))
    sparse_df = sparse.hstack([sparse_df,onehot.fit_transform(np.array(res["item_id"]).reshape(-1,1))])
    sparse_df = sparse.hstack([sparse_df,sparse.csr_matrix(sparse_qmat[np.array(res["item_id"])])])
    sparse_df = sparse.hstack([sparse_df,sparse.csr_matrix(res["wins"])])
    sparse_df = sparse.hstack([sparse_df,sparse.csr_matrix(res["attempts"])])
    
    # Train logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(sparse_df, res["outcome"])
    
    estimated_deltas = model.coef_[0][nb_of_students:nb_of_students+len(true_item_deltas)].reshape(len(true_skill_betas),-1)
    estimated_betas = model.coef_[0][nb_of_students+len(true_item_deltas):nb_of_students+len(true_item_deltas)+len(true_skill_betas)]

    estimated_win_thetas = model.coef_[0][-2*len(true_skill_betas)*5:-len(true_skill_betas)*5].reshape(len(true_skill_betas),-1)
    estimated_attempt_thetas = model.coef_[0][-len(true_skill_betas)*5:].reshape(len(true_skill_betas),-1)
    
    return estimated_deltas, estimated_betas, estimated_win_thetas, estimated_attempt_thetas
