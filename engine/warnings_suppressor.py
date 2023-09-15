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
                        '\n' + '='*80
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
    print(type(all_prints))
    # Remove the welcome
    if BITSANDBYTES_WELCOME in all_prints:
        # print('yes')
        all_prints.replace(BITSANDBYTES_WELCOME, '', 1)
    # print(all_prints)
        print(repr(BITSANDBYTES_WELCOME))
        print(repr(all_prints))