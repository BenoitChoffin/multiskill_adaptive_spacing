import numpy as np
import pandas as pd
from generation import generate_q_matrix, estimate_das3h_params
from student_simulator import single_student
from utils import store_results, print_run_perfs, print_overall_perfs, \
     create_distrib_attempts_dico, create_temp_recall_probas_dico, create_verbose_dico
import os
import json

def das3h_multiskill_simul(B, nb_of_runs, nb_of_students, items_per_skill, list_of_ri,
                           list_of_win_params, list_of_att_params, review_strategies, item_strategies, 
                           reviews_per_step, res_folder, file_name, verbose=True, save_res=False,
                           das3h_1p=False, max_add_train_skills=2, sampling_zone="1"):
    """
    Input:
    * B : number of skills to learn;
    * nb_of_runs: total number of simulation runs;
    * nb_of_students: total number of simulated students (per run);
    * items_per_skill : number of simulated items by skill (cf. generations.generate_q_matrix);
    * list_of_ri : list of retention intervals (in WEEKS);
    * list_of_win_params, list_of_att_params: lists of arrays, each array contains the (ordered)
    odd and even theta parameters. The elements of the arrays are ordered by increasing order of
    time window length.
    * review_strategies: dictionary of skill selection strategies, each of the strategies is the key,
    and as values we have the list of chosen parameters for that strategy;
    * item_strategies: list of chosen item strategies;
    * review_per_step: number of review items that are selected at each learning step (each week);
    * res_folder: name of the folder in which we save the experimental results;
    * file_name: name of the main results file (must end with .csv);
    * save_res: if True, save results;
    * das3h_1p: if True, use only one set of curve parameters for all skills;
    * max_add_train_skills: maximum number of additional skill tags by item;
    * sampling zone: zone in which to sample the theta parameters (cf. IJAIED paper)
    """
    # Initialize results dataframes
    main_df = pd.DataFrame.from_dict({"item_strategy":[],"skill_strategy":[],"param":[],"stud_id":[],
                                      "period":[], "run_id":[], "mean_recall":[],"relative_mean_recall":[],
                                      "sum_attempts":[]})
    temporal_df = pd.DataFrame.from_dict({"item_strategy":[],"skill_strategy":[],"param":[],
                                          "t":[], "run_id":[], "mean_recall":[],"relative_mean_recall":[]})
    # Build distribution attempts dictionary
    review_strats_w_random = review_strategies.copy()
    review_strats_w_random["random_review"] = [0]
    review_strats_w_noreview = review_strats_w_random.copy()
    review_strats_w_noreview["no_review"] = [0]
    distrib_attempts = create_distrib_attempts_dico(review_strats_w_random, item_strategies, B)

    # Initialize main recall probas dictionary
    main_recall_probas_dico = {"run_id":[],"skill_strategy":[],"param":[],"item_strategy":[],"period":[]}
    for b in range(B):
      main_recall_probas_dico["recall_prob_"+str(b)] = []

    for run_id in range(nb_of_runs):
      results = {"skill_strategy":[],"param":[],"run_id":[],"stud_id":[],"t":[],"mean_recall":[],
                 "relative_mean_recall":[],"sum_attempts":[],"item_strategy":[]} # dict for main_df
      if verbose:
        print("Run ",run_id+1)
      verbose_dico = create_verbose_dico(review_strats_w_noreview, item_strategies)
      # Select set of DAS3H curves
      if das3h_1p: # DAS3H_1p = same setting as Khajah et al.
        curve_ids = np.repeat(np.random.choice(len(list_of_win_params),1)[0],B)
      else:
        curve_ids = np.random.choice(len(list_of_win_params),B)
      # Generate student, item and skill params
      student_alphas = np.random.normal(loc=0,scale=1,size=nb_of_students)
      item_deltas = np.random.normal(loc=-1,scale=1,size=(B,items_per_skill))
      skill_betas = np.random.normal(loc=-1,scale=1,size=B)
      qmat, inv_qmat = generate_q_matrix(B, items_per_skill, max_add_train_skills)
      est_deltas, est_betas, est_win_thetas, est_attempt_thetas = estimate_das3h_params(item_deltas.flatten(), skill_betas,
                                                                                        list_of_win_params[curve_ids],
                                                                                        list_of_att_params[curve_ids],
                                                                                        qmat, 200, 10, 3)
      # Build temp recall probas dico
      temp_probas_dico = create_temp_recall_probas_dico(review_strats_w_noreview, item_strategies)

      for stud_id, stud_alpha in enumerate(student_alphas):
        # By default, we always have the 'no review' and 'random review' skill selection strategies
        rand_review_recall_probs = {item_strat_name:[] for item_strat_name in item_strategies}
        for review_strat in ["random_review", "no_review"]:
          # Caution: the two strategies must be in *this* order
          for item_strategy in item_strategies:
            simul_student = single_student(review_strat, 0, B, list_of_ri, stud_alpha, skill_betas,
                                           list_of_win_params[curve_ids], list_of_att_params[curve_ids],
                                           qmat, inv_qmat, item_deltas, items_per_skill,
                                           est_deltas, est_betas, est_win_thetas,
                                           est_attempt_thetas, reviews_per_step, item_strategy)
            res = simul_student.learn_and_review()
            for i in range(B): # Store results from learning phase
              if review_strat == "random_review":
                rand_review_recall_probs[item_strategy].append(res[0][i])
              store_results(results, distrib_attempts, temp_probas_dico, verbose_dico, review_strat, 0, run_id, stud_id, i, res[0][i], res[1][i], res[2][i],
                            (res[0][i]-rand_review_recall_probs[item_strategy][i])/rand_review_recall_probs[item_strategy][i], res[3][i], B, item_strategy)
            for i, ri in enumerate(list_of_ri):
              res = simul_student.get_performance_metric(B-1+ri)
              if review_strat == "random_review":
                rand_review_recall_probs[item_strategy].append(res[0])
              store_results(results, distrib_attempts, temp_probas_dico, verbose_dico, review_strat, 0, run_id, stud_id, B-1+ri, res[0], -1, res[1],
                            (res[0]-rand_review_recall_probs[item_strategy][B+i])/rand_review_recall_probs[item_strategy][B+i], -1*np.ones(len(res[1])), B,
                            item_strategy)
        for review_strat in review_strategies:
          for param in review_strategies[review_strat]:
            for item_strategy in item_strategies:
              simul_student = single_student(review_strat, param, B, list_of_ri, stud_alpha, skill_betas,
                                             list_of_win_params[curve_ids], list_of_att_params[curve_ids],
                                             qmat, inv_qmat, item_deltas, items_per_skill, est_deltas, est_betas,
                                             est_win_thetas, est_attempt_thetas, reviews_per_step, item_strategy)
              res = simul_student.learn_and_review()
              for i in range(B):
                store_results(results, distrib_attempts, temp_probas_dico, verbose_dico, review_strat, param, run_id, stud_id, i, res[0][i], res[1][i], res[2][i],
                              (res[0][i]-rand_review_recall_probs[item_strategy][i])/rand_review_recall_probs[item_strategy][i], res[3][i], B, item_strategy)
              for i, ri in enumerate(list_of_ri):
                res = simul_student.get_performance_metric(B-1+ri)
                store_results(results, distrib_attempts, temp_probas_dico, verbose_dico, review_strat, param, run_id, stud_id, B-1+ri, res[0], -1, res[1],
                              (res[0]-rand_review_recall_probs[item_strategy][B+i])/rand_review_recall_probs[item_strategy][B+i], -1*np.ones(len(res[1])), B,
                              item_strategy)
      if verbose:
        print_run_perfs(verbose_dico)
      # Average correctness probas from temp_probas_dico and store them inside main_recall_probas_dico
      for strat_name in review_strats_w_noreview.keys():
        for param_value in review_strats_w_noreview[strat_name]:
          for item_strat in item_strategies:
            for period in ["learning","retention"]:
              main_recall_probas_dico["run_id"].append(run_id)
              main_recall_probas_dico["skill_strategy"].append(strat_name)
              main_recall_probas_dico["param"].append(param_value)
              main_recall_probas_dico["item_strategy"].append(item_strat)
              main_recall_probas_dico["period"].append(period)
              for b in range(B):
                main_recall_probas_dico["recall_prob_"+str(b)].append(np.around(np.array(temp_probas_dico[strat_name][param_value][item_strat][period])[:,b].mean(),5))
      # Compute averaged results within the run
      results_df = pd.DataFrame.from_dict(results)
      results_df['period'] = results_df['t'].apply(lambda x: "learning" if x < B else "retention")
      results_df['run_id'] = run_id
      main_df = pd.concat([main_df,results_df.groupby(["item_strategy","skill_strategy",
        "param","stud_id","period","run_id"],as_index=False).agg({"mean_recall":lambda x: np.around(x.mean(),5),
        "relative_mean_recall":lambda x: np.around(x.mean(),5),"sum_attempts":"max"})])
      temporal_df = pd.concat([temporal_df,results_df.groupby(["item_strategy","skill_strategy",
        "param","t","run_id"],as_index=False).agg({"mean_recall":lambda x: np.around(x.mean(),5),
        "relative_mean_recall":lambda x: np.around(x.mean(),5)})])

    if verbose:
      print_overall_perfs(main_df)

    recall_probas_df = pd.DataFrame.from_dict(main_recall_probas_dico)

    # Build attempts df
    attempts_df = {"skill_strategy":[],"param":[],"item_strategy":[],"t":[]}
    for b in range(B):
      attempts_df["att_skill_"+str(b)] = []
    for skill_strat, v1 in distrib_attempts.items():
      for param, v2 in v1.items():
        for item_strat, v3 in v2.items():
          for t, v4 in v3.items():
            attempts_df["skill_strategy"].append(skill_strat)
            attempts_df["param"].append(param)
            attempts_df["item_strategy"].append(item_strat)
            attempts_df["t"].append(t)
            for b in range(B):
              attempts_df["att_skill_"+str(b)].append(v4["att_skill_"+str(b)])
    attempts_df = pd.DataFrame(attempts_df)

    # Modify column types + reset index
    main_df.reset_index(inplace=True, drop=True)
    temporal_df.reset_index(inplace=True, drop=True)
    main_df["run_id"] = main_df["run_id"].astype(np.int32)
    main_df["stud_id"] = main_df["stud_id"].astype(np.int32)
    attempts_df["t"] = attempts_df["t"].astype(np.int32)
    temporal_df["t"] = temporal_df["t"].astype(np.int32)
    temporal_df["run_id"] = temporal_df["run_id"].astype(np.int32)
    for j in range(B):
      attempts_df["att_skill_"+str(j)] = attempts_df["att_skill_"+str(j)].astype(np.int32)
    if save_res:
      main_df.to_csv(os.path.join(res_folder,file_name), index=False)
      temporal_df.to_csv(os.path.join(res_folder,"temporal.csv"), index=False)
      attempts_df.to_csv(os.path.join(res_folder,"attempts.csv"), index=False)
      recall_probas_df.to_csv(os.path.join(res_folder,"recall_probas.csv"), index=False)
      serializable_review_strats = {k:list(np.around(v,2).astype(float)) for k,v in review_strategies.items()}
      with open(os.path.join(res_folder, 'metadata.json'), 'w') as f:
        f.write(json.dumps({
          'review_strategies': serializable_review_strats,
          'nb_of_runs': nb_of_runs,
          'nb_of_students': nb_of_students,
          'items_per_skill':items_per_skill,
          'item_strategies':item_strategies,
          'reviews_per_step':reviews_per_step,
          'das3h_1p':das3h_1p,
          'max_add_train_skills':max_add_train_skills,
          'sampling_zone':sampling_zone
          }, indent=4))
    return main_df, temporal_df, attempts_df, recall_probas_df
