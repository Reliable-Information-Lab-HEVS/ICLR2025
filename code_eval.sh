docker build -t code_sandbox -f code_execution/Dockerfile .
docker run -it --name my_container code_sandbox
docker cp my_container:/LLMs/results/HumanEval_results results/HumanEval_results
docker rm my_container
docker rmi code_sandbox