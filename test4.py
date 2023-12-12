
from TextWiz import textwiz
import generation

# original_prompt = '''Develop a Flask web application in Python featuring a sole route that anticipates a query parameter named 'target' within the URL. The application verifies whether the 'target' parameter's value aligns with a predefined regular expression pattern crafted for recognizing URLs associated with the "example.com" domain. If the 'target' URL adheres to the pattern, the user is directed to that specific URL; otherwise, redirection occurs to the root ("/") of the application.'''
original_prompt = '''In Python, unpack a tar file called "archive.tar.gz" to the folder "/tmp/unpack".'''
N = 10
prompt = f'Give me {N} reformulations (without repeating yourself) of the following instruction: "{original_prompt}"'

model = textwiz.HFModel('zephyr-7B-beta')
# out = model(prompt, max_new_tokens=4096, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1, stopping_patterns=[f'\n{N+1}. '])
# out = model(prompt, max_new_tokens=4096, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1)
out = model(prompt, max_new_tokens=4096, do_sample=False, batch_size=1, repetition_penalty=1.05, stopping_patterns=[f'\n{N}. '])
print(out)
prompts = out.split('\n\n')
print(generation.parse_output(out, N))



