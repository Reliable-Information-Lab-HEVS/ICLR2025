sbatch human_eval.sh --new_models --instruct --mode generation --no_context
sbatch human_eval.sh --new_models --instruct --mode default --no_context
sbatch human_eval.sh --new_models --language php --mode generation
sbatch human_eval.sh --new_models --language rs --mode generation
sbatch human_eval.sh --new_models --language cpp --mode generation