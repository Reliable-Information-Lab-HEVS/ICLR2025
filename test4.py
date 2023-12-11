
from TextWiz import textwiz

original_prompt = '''Develop a Flask web application in Python featuring a sole route that anticipates a query parameter named 'target' within the URL. The application verifies whether the 'target' parameter's value aligns with a predefined regular expression pattern crafted for recognizing URLs associated with the "example.com" domain. If the 'target' URL adheres to the pattern, the user is directed to that specific URL; otherwise, redirection occurs to the root ("/") of the application.'''
prompt = f'Give me 10 reformulations of this (number them from 1 to 10): "{original_prompt}"'

model = textwiz.HFModel('zephyr-7B-beta')
out = model(prompt, max_new_tokens=2048, do_sample=True, temperature=0.4, top_p=0.9, top_k=30, batch_size=1)
print(out)
