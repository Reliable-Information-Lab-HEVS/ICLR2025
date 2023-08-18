docker build -t code_sandbox -f code_execution/Dockerfile .
docker run -it --name my_container code_sandbox
docker cp my_container:/LLMs/results/HumanEvalInstruct_completions_results results/HumanEvalInstruct_completions_results
docker rm my_container
docker rmi code_sandbox