FROM python:3.11.3

# Set user to non-root to avoid potential issues when running arbitrary code inside the container
RUN groupadd -r random_group && useradd --no-log-init -r -g random_group random_user
USER random_user

WORKDIR /LLMs

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

COPY results/HumanEval_completions results/HumanEval_completions
COPY code_execution code_execution

CMD [ "python", "code_execution/test.py" ]