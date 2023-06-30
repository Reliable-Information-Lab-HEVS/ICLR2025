import torch
import numpy as np
import argparse
import time

import loader
import engine
import utils
import agents

from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain

llm = agents.HuggingFaceLLM.from_name('star-coder', max_new_tokens=300)
tools = [agents.Flake8Tool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Write code to multiply 2 numbers, then refactore it according to Flake8.")
