import warnings
import logging
import contextlib
import io

class LoggingFilter(logging.Filter):
    """Used to remove messages or warnings issued by the `logging` library.
    """

    def __init__(self, patterns: list[str] | str):

        super().__init__()
        self.patterns = [patterns] if type(patterns) == str else patterns

    def filter(self, record):
        return not any(pattern in record.getMessage() for pattern in self.patterns)
    

BETTER_TRANSFORMER_WARNING = ('The BetterTransformer implementation does not support padding during training, '
                              'as the fused kernels do not support attention masks. Beware that passing padded '
                              'batched data during training may result in unexpected outputs.')

optimum_logger = logging.getLogger('optimum.bettertransformer.transformation')
optimum_logger.addFilter(LoggingFilter(BETTER_TRANSFORMER_WARNING))


# warnings.filterwarnings(action='ignore', message=better_transformer_warning)


BITSANDBYTES_WELCOME = '\n' + '='*35 + 'BUG REPORT' + '='*35 + '\n' + \
                        ('Welcome to bitsandbytes. For bug reports, please run\n\npython -m bitsandbytes\n\n'
                        ' and submit this information together with your error trace to: '
                        'https://github.com/TimDettmers/bitsandbytes/issues') + \
                        '\n' + '='*80 + '\n'
BITSANDBYTES_SETUPS = (
    'CUDA SETUP: CUDA runtime path found:',
    'CUDA SETUP: Highest compute capability among GPUs detected:',
    'CUDA SETUP: Detected CUDA version',
    'CUDA SETUP: Loading binary',
    )

@contextlib.contextmanager
def swallow_bitsandbytes_prints():
    """Remove all prints when bitsandbytes is initialized correctly (they cluter the output stream, especially
    when doing multiprocessing).
    """
    with contextlib.redirect_stdout(io.StringIO()) as f:
        yield
    all_prints = f.getvalue()
    # Remove all bitsandbytes setup output
    if BITSANDBYTES_WELCOME in all_prints:
        to_keep, after = all_prints.split(BITSANDBYTES_WELCOME, 1)
        lines = after.splitlines()

        # first line after the welcome is just a path
        lines_to_keep = []
        for i, line in enumerate(lines[1:]):
            if i < len(BITSANDBYTES_SETUPS):
                assert line.startswith(BITSANDBYTES_SETUPS[i]), 'The bitsandbytes print format is not as exepected'
            else:
                lines_to_keep.append(line)
        
        to_keep += '\n'.join(lines_to_keep)

        # reprint without the bitsandbytes block
        if to_keep != '':
            print(to_keep)

    else:
        if all_prints != '':
            print(all_prints)
    


        