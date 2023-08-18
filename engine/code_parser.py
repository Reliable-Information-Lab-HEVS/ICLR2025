import re

PYTHON_START_KEYWORDS = [
    '\nfrom ',
    '\nimport ',
    '\ndef ',
    '\n# ',
    '\nclass ',
    '\n@',
    '\nif ',
    '\nfor ',
    '\nwhile '
    '\nprint(',
]

# group that matches = symbol ???? NOT WORKING
PYTHON_LINE = r'(?:\n|^)(?:.*?)=(?:.*?)(?:\n|$)'

START_KEYWORDS_GROUP = r'(?:' + '|'.join(map(re.escape, PYTHON_START_KEYWORDS)) + r')'


test_regex = START_KEYWORDS_GROUP + r'(.*?)' + r'(?:(?!' + START_KEYWORDS_GROUP + r'))'
# regex that matches any start keyword
# regex that matches any word after a breakline that IS NOT a start keyword
end_regex = r'(?:?!' + START_KEYWORDS_REGEX + r')'


CODE_REGEX = [
    # Regexes that match the usual markdown python syntax with 3 backticks
    r'```python\n(.*?)\n```',
    r'```\n(.*?)\n```'

    # 
    r"Here's(?:.*?)(?:implementation|implement)(?:.*?):\n",
    r'\b(?:' + START_KEYWORDS_REGEX + r')\b'
]


# USE RE.DOTALL in match !!!!!!!!