docker build -t code_sandbox -f code_execution/Dockerfile .
docker run code_sandbox
# docker run -it --entrypoint bash code_sandbox