import numpy as np
from collections import defaultdict
from strategies import greedy_skill_selection, greedy_multiskill_selection
from utils import gen_outcome, intersection, custom_argmax, custom_argmin,\
     gen_cplx_outcome, OurQueue

class single_student:
    """
    Class for simulating a single student, based on the DAS3H model.

    Input:
        * review_strat: name of the skill selection strategy;
        * param_review: parameter chosen for the skill selection strategy;
        * B: number of blocks (skills) in the learning simulation;
        * list_of_ri: list of retention intervals. Each ri in list_of_ri indicates that the student's proficiency will
        be assessed at timestamp B-1+ri;
        * stud_alpha: alpha (ability) parameter of the student;
        * skill_betas: list of beta easiness parameters, in the same order as the introduction of each skill;
        * list_of_win_params: list of arrays. Each array contains the odd (wins) thetas for each skill b. Inside
        each thetas array, the thetas are ordered from the shortest to the largest time window.
        * list_of_att_params: list of arrays. Each array contains the even (attempts) thetas for each skill b. Inside
        each thetas array, the thetas are ordered from the shortest to the largest time window.
        * q_mat: q-matrix, same as output from generation.generate_q_matrix;
        * inv_q_mat: inverse q-matrix, same as output from generation.generate_q_matrix;
        * item_deltas: array of item deltas, ordered as in the q-matrix;
        * items_per_skill: number (int) of items to generate per skill (cf. generation.generate_q_matrix);
        * est_deltas: estimated deltas (for the estimated version of the skill strategies);
        * est_betas: estimated betas (for the estimated version of the skill strategies);
        * est_win_params, est_att_params: estimated win and att params (for the estimated version of the skill strategies);
        * reviews_per_step: int, number of reviews performed by a student at each week;
        * item_strat: str, item strategy name, either 'random' or 'max_skills'
    """
    def __init__(self, review_strat, param_review, B, list_of_ri, stud_alpha, skill_betas, list_of_win_params,
                 list_of_att_params, q_mat, inv_q_mat, item_deltas, items_per_skill,
                 est_deltas, est_betas, est_win_params, est_att_params, reviews_per_step=3, item_strat="random"):
        self.review_strat = review_strat
        self.param_review = param_review
        self.B = B
        self.list_of_ri = list_of_ri
        self.stud_alpha = stud_alpha
        self.skill_betas = skill_betas
        self.list_of_win_params = list_of_win_params # Attention, must be the same as the ones chosen by the algo!
        self.list_of_att_params = list_of_att_params # Attention, must be the same as the ones chosen by the algo!
        self.qmat = q_mat
        self.inv_q_mat = inv_q_mat
        self.item_deltas = item_deltas
        self.items_per_skill = items_per_skill
        self.est_deltas = est_deltas
        self.est_betas = est_betas
        self.est_win_params = est_win_params
        self.est_att_params = est_att_params
        self.reviews_per_step = reviews_per_step
        self.item_strat = item_strat
        self.q = defaultdict(lambda: OurQueue())
    
    def choose_skill(self, week):
        """
        Choose skill to review on a given week.

        Input:
            * week: int, current week number. Starts in the beginning at 0.
        """
        if self.review_strat == "random_review":
            selected_block = np.random.choice(week,1)[0]
        elif self.review_strat == "mu_back": # mu must be > 0
            selected_block = max(week-self.param_review,0)
        elif self.review_strat.startswith("theta_thres"):
            # Output for uniskill strats = skill id; for multiskill strats = item id
            probas_recall = []
            for j in range(week):
                win_counters = self.q[(j,"wins")].get_counters(week*3600*24*7)
                attempt_counters = self.q[(j,"attempts")].get_counters(week*3600*24*7)
                if self.review_strat in ["theta_thres","theta_thres_multiskill"]:
                    probas_recall.append(gen_outcome(self.stud_alpha, self.item_deltas.mean(), self.skill_betas[j],
                                                     win_counters, attempt_counters, self.list_of_win_params[j],
                                                     self.list_of_att_params[j]))
                elif self.review_strat in ["theta_thres_est","theta_thres_multiskill_est"]:
                    # We put 0 as the mean ability for the students
                    probas_recall.append(gen_outcome(0, self.est_deltas.mean(), self.est_betas[j],
                                                     win_counters, attempt_counters, self.est_win_params[j],
                                                     self.est_att_params[j]))
            if self.review_strat in ["theta_thres","theta_thres_est"]:
                selected_block = np.argmin(np.absolute(np.array(probas_recall)-self.param_review))
            elif self.review_strat in ["theta_thres_multiskill","theta_thres_multiskill_est"]:
                selected_skills = []
                selected_skills.append(np.argmin(np.absolute(np.array(probas_recall)-self.param_review)))
                # Then, selection of the other skills
                # We need to loop over all possible skills, in the decreasing order of proximity with theta
                increasing_order = np.argsort(np.absolute(np.array(probas_recall)-self.param_review))
                # We need to define a list of acceptable items that do *not* involve skills seen after the current week (included)
                acceptable_items = [item_id for item_id in self.inv_q_mat[selected_skills[0]] if item_id < week*self.items_per_skill]
                for k in range(1,len(probas_recall)):
                    intersec_items = intersection(acceptable_items,self.inv_q_mat[increasing_order[k]])
                    if len(intersec_items) == 0:
                        # Goes to next skill if there is no other item that meets the skills criterion
                        continue
                    else:
                        acceptable_items = intersec_items.copy()
                # Choose item
                # There will be at least 1 item inside acceptable_items
                selected_block = np.random.choice(acceptable_items) # We call it "block" but in fact it's the item index
        elif self.review_strat in ["greedy","greedy_est"]:
            # Output for uniskill strats = skill id
            if self.review_strat == "greedy":
                selected_block = greedy_skill_selection(week, self.B, self.list_of_ri, self.stud_alpha, self.item_deltas.mean(),
                                                        self.skill_betas, self.list_of_win_params, self.list_of_att_params, self.q)
            elif self.review_strat == "greedy_est":
                selected_block = greedy_skill_selection(week, self.B, self.list_of_ri, 0, self.est_deltas.mean(), self.est_betas,
                                                        self.est_win_params, self.est_att_params, self.q)
        elif self.review_strat in ["greedy_multiskill","greedy_multiskill_est"]:
            # Output for multiskill strats = item id
            if self.review_strat == "greedy_multiskill":
                selected_block = greedy_multiskill_selection(week, self.B, self.list_of_ri, self.stud_alpha, self.item_deltas.mean(),
                                                             self.skill_betas, self.list_of_win_params, self.list_of_att_params, self.q,
                                                             self.qmat, self.inv_q_mat, self.items_per_skill)
            elif self.review_strat == "greedy_multiskill_est":
                selected_block = greedy_multiskill_selection(week, self.B, self.list_of_ri, 0, self.est_deltas.mean(), self.est_betas,
                                                             self.est_win_params, self.est_att_params, self.q, self.qmat,
                                                             self.inv_q_mat, self.items_per_skill)
        return selected_block
    
    def choose_item(self, selected_block, week):
        acceptable_items = [item_id for item_id in self.inv_q_mat[selected_block] if item_id < week*self.items_per_skill]
        if self.item_strat == "random":
            selected_item = np.random.choice(acceptable_items)
        elif self.item_strat == "max_skills":
            temp_nb_skills = []
            for item_id in acceptable_items:
                temp_nb_skills.append(len(self.qmat[item_id]))
            selected_item = acceptable_items[custom_argmax(np.array(temp_nb_skills))]
        return selected_item
    
    def learn_and_review(self):
        temp_recall_probs = [] # stores lists of correctness probas for each skill, at each learning week
        sum_attempts = [] # stores temp_sum_attempts for each week
        temp_sum_attempts = 0 # counts total nb of skills reviewed up to any timestamp
        temp_results = [] # stores ACP at each learning week
        attempts_per_skill = [] # stores lists of skill review attempts for any week
        for week in range(self.B):
            # Learn weekly material
            self.q[(week,"attempts")].push(week*7*24*3600) # When students learn, we add +1 attempt to their history
            temp_sum_attempts += 1
            # Test : are there attempts that have been pushed for a timestamp > current timestamp (cf. OurQueue doc)?
            current_queue_att = self.q[(week,"attempts")].queue
            current_queue_win = self.q[(week,"wins")].queue
            if len(current_queue_att) > 0:
                if np.max(current_queue_att) > week*7*24*3600:
                    print("Error: the queue contains posterior attempts. Please refer to utils.OurQueue doc.")
            if len(current_queue_win) > 0:
                if np.max(current_queue_win) > week*7*24*3600:
                    print("Error: the queue contains posterior wins. Please refer to utils.OurQueue doc.")
            # A student can only review from 2nd week (= week 1) and can only review blocks until past week
            list_att_per_skill = np.zeros(self.B) # stores skill review attempts in a given week
            if (self.review_strat != "no_review") & (week > 0):
                for j in range(self.reviews_per_step):
                    selected_block = self.choose_skill(week)
                    # Choose item which involves selected skill
                    if self.review_strat in ["greedy_multiskill","greedy_multiskill_est",
                                             "theta_thres_multiskill","theta_thres_multiskill_est"]:
                        selected_item_id = selected_block
                        # Store attempts on each selected skill
                        for b in self.qmat[selected_item_id]:
                            list_att_per_skill[b] += 1
                    else:
                        selected_item_id = self.choose_item(selected_block, week)
                        # Store attempts on the **selected** skill
                        list_att_per_skill[selected_block] += 1
                    selected_skill_indices = self.qmat[selected_item_id]

                    list_win_counters = []
                    list_attempt_counters = []
                    list_beta = []
                    list_h_features = []
                    for b in selected_skill_indices:
                        temp_sum_attempts += 1
                        list_win_counters.append(self.q[(b,"wins")].get_counters(week*7*24*3600))
                        list_attempt_counters.append(self.q[(b,"attempts")].get_counters(week*7*24*3600))
                        list_beta.append(self.skill_betas[b])
                        list_h_features.append((self.list_of_win_params[b],self.list_of_att_params[b]))
                        self.q[(b,"attempts")].push(week*7*24*3600)
                    outcome = gen_cplx_outcome(self.stud_alpha, self.item_deltas.flatten()[selected_item_id],
                                               list_beta, list_win_counters, list_attempt_counters, list_h_features)
                    if outcome > .5: # Deterministic output
                        for b in selected_skill_indices:
                            self.q[(b,"wins")].push(week*7*24*3600)
            res = self.get_performance_metric(week) # We compute the metrics *after* the reviewing phase
            temp_results.append(res[0])
            sum_attempts.append(temp_sum_attempts)
            temp_recall_probs.append(res[1])
            attempts_per_skill.append(list(list_att_per_skill))
        return temp_results, sum_attempts, temp_recall_probs, attempts_per_skill
    
    def get_performance_metric(self, t):
        """
        Return student performance metric (ACP) at a given t timestamp (WEEK).
        """
        temp_recall_probs = []
        for week in range(self.B):
            temp_recall_probs.append(gen_outcome(self.stud_alpha,-1,self.skill_betas[week],
                                                 self.q[(week,"wins")].get_counters(t*7*3600*24),
                                                 self.q[(week,"attempts")].get_counters(t*7*3600*24),
                                                 self.list_of_win_params[week],
                                                 self.list_of_att_params[week]))
        return np.mean(temp_recall_probs), temp_recall_probs
