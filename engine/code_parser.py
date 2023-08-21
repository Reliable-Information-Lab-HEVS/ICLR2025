import re

from engine import stopping

# NOTE: This parser is not perfect! It was designed for a specific use case, and is not foolproof.

# Python keywords suseptible to start a line
PYTHON_START_KEYWORDS = [
    'from ',
    'import ',
    'def ',
    '#',
    'class ',
    '@',
    'if ',
    'for ',
    'while ',
    'print',
]

# Patterns that represent a Python line
PYTHON_PATTERNS = [
    # matches a line containing a "=" symbol, without containing a newline before it
    r'(?:[^\n]+)=',
]

# Regex containing all pythons markers as a non-capturing OR (chained with pipes |) group
PYTHON_MARKERS = '|'.join(map(re.escape, PYTHON_START_KEYWORDS)) + '|' + '|'.join(PYTHON_PATTERNS)
PYTHON_GROUP = r'(?:' + PYTHON_MARKERS + r')'

# matches anything between start of string or newline followed by a PYTHON_MARKER, and newline followed
# by text (not space) that is not a PYTHON_MARKER, or end of string
# \nPYTHON_MARKER (blabla) \nNON-PYTHON_MARKER
# NOTE: standalone statements such as "foo.append(1)" which are not indented (inside a block such as if, for etc)
# will not be recognized
PYTHON_CODE_REGEX = r'(?:^|\n)(' + PYTHON_GROUP + r'.*?)' + r'(?:$|(?:\n(?!' + PYTHON_GROUP + r')\S))'

# Regexes suseptible to capture python code. Starting with the easiest and simplest form of blocks to
# parse, and increasing in complexity (and thus possible parsing errors)
# NOTE: Every regex needs to be searched with the flag re.DOTALL
PYTHON_CODE_REGEXES = [
    # Regexes that match the usual markdown python syntax with 3 backticks
    r'```python\n(.*?)(?:$|\n```)',
    r'```\n(.*?)(?:$|\n```)',

    PYTHON_CODE_REGEX,
]


class PythonParser(object):
    """Parser that extracts Python code from strings.

    NOTE: It is not perfect! The following patterns are NOT parsed correctly:

    - text containing code

    """

    def __init__(self):
        self.python_start_keywords = PYTHON_START_KEYWORDS
        self.python_patterns = PYTHON_PATTERNS
        self.code_regexes = PYTHON_CODE_REGEXES

    
    def __call__(self, s: str, stopping_patterns: list[str] | tuple[str] | None = None) -> str:
        """Parse all Python code contained in `s`, concatenate it, and truncate it according to the 
        patterns in `stopping_patterns`.

        Parameters
        ----------
        s : str
            String to parse.
        stopping_patterns : list[str] | tuple[str] | None, optional
            Patterns on which to truncate the result, by default None

        Returns
        -------
        str
            The truncated code output.
        """
        return self.full_parse(s, stopping_patterns)
    

    def parse(self, s: str) -> list[str]:
        """Parse a string `s`, and return all Python code blocks withon `s`. This function try to parse `s`
        with methods of increasing complexity. As soon, as one method produces at least one match, we consider
        the parsing as succesful and return that match.

        Parameters
        ----------
        s : str
            String to parse.

        Returns
        -------
        list[str]
            List containing all code blocks.
        """

        for regex in self.code_regexes:
            matches = re.findall(regex, s, re.DOTALL)
            if len(matches) > 0:
                # remove trailing (and leading, but there should not be any) newlines
                return [x.strip() for x in matches]
            

    def concatenate(self, code_blocks: list[str]) -> str:
        """Concatenate multiple code blocks into a single code block.

        Parameters
        ----------
        code_blocks : list[str]
            The strings representing the code blocks

        Returns
        -------
        str
            Single code block.
        """
            
        return '\n\n'.join(code_blocks)


    def truncate(self, code_block: str, stopping_patterns: list[str] | tuple[str] | None = None) -> str:
        """Truncate `code_block` as soon as one of `stopping_patterns` is found.

        Parameters
        ----------
        code_block : str
            The code block to possibly truncate.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to scan for, by default None

        Returns
        -------
        str
            The truncated `code_block`.
        """
        
        truncated = stopping.post_process_stopping_patterns([code_block], stopping_patterns)
        return truncated[0]
    

    def full_parse(self, s: str, stopping_patterns: list[str] | tuple[str] | None = None) -> str:
        """Parse all Python code contained in `s`, concatenate it, and truncate it according to the 
        patterns in `stopping_patterns`.

        Parameters
        ----------
        s : str
            String to parse.
        stopping_patterns : list[str] | tuple[str] | None, optional
            Patterns on which to truncate the result, by default None

        Returns
        -------
        str
            The truncated code output.
        """

        blocks = self.parse(s)
        block = self.concatenate(blocks)
        truncated_block = self.truncate(block, stopping_patterns)
        return truncated_block
    



# Some simple tests (because parsing code is very prone to errors)

_TEST_INPUTS = [
    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)
```""",

    """Here's a Python implementation of the function:
```python
def foo(bar):
    print(bar)""",

"""def foo(bar):
    print(bar)""",

    """Here's a Python implementation of the function:
```python
def foo(bar):
    baz = \"\"\"String with backticks:```
    foo
    ```\"\"\"
    print(bar)
```""",

    # Without backticks
    """Here's a Python implementation of the function:

def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations
    
Here's an example usage:

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",
]


_EXPECTED_OUTPUTS = [
    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    print(bar)""",

    """def foo(bar):
    baz = \"\"\"String with backticks:```
    foo
    ```\"\"\"
    print(bar)"""

    """def parse_music(music_string: str) -> List[int]:
    \"\"\"Parses a music string in the special ASCII format and returns a list of note durations.\"\"\"
    note_durations = []
    current_duration = 0
    for char in music_string:
        if char == "o":
            current_duration = 4
        elif char == "o|":
            current_duration = 2
        elif char == ".|":
            current_duration = 1
        else:
            raise ValueError(f"Invalid character in music string: {char}")
        note_durations.append(current_duration)
    return note_durations

music_string = "o o|.| o| o|.|.|.|.| o o"
note_durations = parse_music(music_string)
print(note_durations) # Output: [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]""",

]


def _run_tests():

    parser = PythonParser()

    for i, (input, output) in enumerate(zip(_TEST_INPUTS, _EXPECTED_OUTPUTS)):
        assert parser(input) == output, f'test {i} failed'

    print('All tests completed succesfully')