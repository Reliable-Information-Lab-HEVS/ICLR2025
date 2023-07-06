import torch
import numpy as np
import argparse
import time

from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain

import engine
from helpers import utils

# llm = agents.HuggingFaceLLM.from_name('star-coder', max_new_tokens=300)
# tools = [agents.Flake8Tool()]
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("Write code to multiply 2 numbers, then refactore it according to Flake8.")

prompt = "Write code to multiply 2 numbers"
t0 = time.time()
model = engine.HFModel('star-coder')
dt = time.time() - t0
print(f'Time to load the model: {dt:.2f} s')

t1 = time.time()
res = model(prompt)
dt1 = time.time() - t1
print(f'Time for inference: {dt1:.2f} s')
print(f'Output: {res}')

