docker build -t code_sandbox -f code_execution/Dockerfile .
docker run -it --name my_container code_sandbox

# This is used to copy back the results
python3 code_execution/file_extraction.py

docker rm my_container
docker rmi code_sandbox