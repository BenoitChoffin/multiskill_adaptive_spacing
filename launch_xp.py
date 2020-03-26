import argparse
import numpy as np
from generation import gen_random_das3h_curve
from experiment import das3h_multiskill_simul
from utils import prepare_folder, build_new_paths
import os
import json

parser = argparse.ArgumentParser(description='Run multiskill item selection simulations.')
parser.add_argument('res_folder', type=str, nargs='?')
parser.add_argument('file_name', type=str, nargs='?', default="results.csv")
parser.add_argument('--B', type=int, nargs='?', default=10)
parser.add_argument('--nb_runs', type=int, nargs='?')
parser.add_argument('--nb_students', type=int, nargs='?')
parser.add_argument('--train_items_per_skill', type=int, nargs='?', default=20)
parser.add_argument('--ri_max', type=int, nargs='?', default=6)
parser.add_argument('--item_strategy', type=str, nargs='?', default="all")
parser.add_argument('--reviews_per_step', type=int, nargs='?', default=3)
parser.add_argument('--verbose', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--save_res', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--das3h_1p', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--max_add_train_skills', type=int, nargs='?', default=2)
parser.add_argument('--sampling_zone', type=str, nargs='?', default='1')
options = parser.parse_args()

experiment_args = vars(options)

RES_PATH = './results/'+options.res_folder
prepare_folder(RES_PATH)

# Get maximum id of folders
if options.save_res:
  if os.path.isfile(os.path.join(RES_PATH,"folder_data.json")):
    with open(os.path.join(RES_PATH,"folder_data.json"), 'r') as f:
      data = json.load(f)
      data['max_folder'] += 1
      MAX_FOLDER = data['max_folder']
      os.remove(os.path.join(RES_PATH,"folder_data.json"))
    with open(os.path.join(RES_PATH,"folder_data.json"), 'w') as f:
      json.dump(data, f, indent=4)
  else:
    with open(os.path.join(RES_PATH,"folder_data.json"), 'w') as f:
      f.write(json.dumps({"max_folder":0}, indent=4))
      MAX_FOLDER = 0
  FULL_RES_PATH = os.path.join(RES_PATH,str(MAX_FOLDER))
  prepare_folder(FULL_RES_PATH)
else:
  FULL_RES_PATH = None

review_strategies = {"mu_back":[1,2,3],
                     "theta_thres":np.linspace(0,1,11),
                     "theta_thres_est":np.linspace(0,1,11),
                     "theta_thres_multiskill":np.linspace(0,1,11),
                     "theta_thres_multiskill_est":np.linspace(0,1,11),
                     "greedy":[0],
                     "greedy_est":[0],
                     "greedy_multiskill":[0],
                     "greedy_multiskill_est":[0]}

if options.item_strategy == "all":
  item_strategies = ["random", "max_skills"]
elif options.item_strategy == "max_skills":
  item_strategies = ["max_skills"]
else:
  item_strategies = ["random"]

das3h_curves = gen_random_das3h_curve(1000, zone=options.sampling_zone)
expe = das3h_multiskill_simul(options.B, options.nb_runs, options.nb_students,
                              options.train_items_per_skill, np.arange(1,options.ri_max+1), das3h_curves[0],
                              das3h_curves[1], review_strategies, item_strategies, options.reviews_per_step,
                              FULL_RES_PATH, options.file_name, options.verbose, options.save_res,
                              options.das3h_1p, options.max_add_train_skills, options.sampling_zone)
