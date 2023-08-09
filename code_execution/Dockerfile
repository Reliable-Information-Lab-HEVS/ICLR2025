FROM python:3.11.3

WORKDIR /LLMs

COPY results/HumanEval_completions results/HumanEval_completions
COPY data/HumanEval.jsonl data/HumanEval.jsonl
COPY code_execution code_execution
COPY helpers helpers
COPY code_eval.py code_eval.py

RUN pip install --no-cache-dir -r code_execution/requirements.txt

# Set user to non-root to avoid potential issues when running arbitrary code inside the container
RUN groupadd -r random_group && useradd --no-log-init -r -g random_group random_user
USER random_user

CMD [ "python3", "code_eval.py" ]
# CMD [ "python3", "code_execution/test.py" ]