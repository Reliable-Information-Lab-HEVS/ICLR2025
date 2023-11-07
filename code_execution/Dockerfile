FROM python:3.11.3

# Install other programming languages
RUN apt-get update
RUN apt-get install php -y
RUN apt-get install g++ -y
RUN apt-get install rustc -y


WORKDIR /LLMs

COPY results results
COPY data data
COPY code_execution code_execution
COPY helpers helpers
COPY code_eval.py code_eval.py

RUN pip install --no-cache-dir -r code_execution/requirements.txt

# Set user to non-root to avoid potential issues when running arbitrary code inside the container
RUN groupadd -r random_group && useradd --no-log-init -r -g random_group random_user
# Give user ownership of the directory so that it has write permission to save the results
RUN chown -R random_user /LLMs
USER random_user

CMD [ "python3", "code_eval.py" ]