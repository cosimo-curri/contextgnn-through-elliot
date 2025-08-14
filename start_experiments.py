import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--model', type=str, default='contextgnn')
args = parser.parse_args()

run_experiment(f"config_files/{args.model}_{args.dataset}.yml")
