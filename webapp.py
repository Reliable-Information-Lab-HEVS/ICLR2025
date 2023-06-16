import gradio as gr
import gc

import loader
import utils


# Default model to load at start-up
DEFAULT = 'bloom-560M'

# File where the valid credentials are stored
CREDENTIALS_FILE = utils.ROOT_FOLDER + '/.gradio_login.txt'

# Initialize global model and tokenizer (necessary not to reload the model for each new inference)
model, tokenizer = loader.load_model_and_tokenizer(DEFAULT)


def update_model(model_name: str):
    """Update the model and tokenizer in the global scope so that we can reuse it and speed up inference.

    Parameters
    ----------
    model_name : str
        The name of the new model to use.
    """
    
    global model
    global tokenizer
    del model
    del tokenizer
    gc.collect()
    model = loader.load_model(model_name)
    tokenizer = loader.load_tokenizer(model_name)


def text_generation(prompt: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 100,
                    top_p: float = 0.92, temperature: float = 0.9, num_return_sequences: int = 1,
                    use_seed: bool = False, seed: int | None = None):
    """Text generation using the model and tokenizer in the global scope, so that we can reuse them for multiple
    prompts.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.
    do_sample : bool, optional
        Whether to introduce randomness in the generation, by default True.
    top_k : int, optional
        How many tokens with max probability to consider for randomness, by default 100.
    top_p : float, optional
        The probability density covering the new tokens to consider for randomness, by default 0.92.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness), by default 0.9.
    num_return_sequences : int, optional
        How many sequences to generate according to the `prompt`, by default 1.
    use_seed : bool, optional
        Whether to use a fixed seed for reproducibility., by default False.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.
    Returns
    -------
    list[str]
        List containing all `num_return_sequences` sequences generated.
    """
    
    if not use_seed:
        seed = None
    predictions = loader.generate_text(model, tokenizer, prompt, max_new_tokens, do_sample, top_k, top_p,
                                       temperature, num_return_sequences, seed)
    return utils.format_output(predictions)


def authentication(username: str, password: str) -> bool:
    """Simple authentication method.

    Parameters
    ----------
    username : str
        The username provided.
    password : str
        The password provided.

    Returns
    -------
    bool
        Return True if both the username and password match the credentials stored in `CREDENTIALS_FILE`. 
        False otherwise.
    """

    with open(CREDENTIALS_FILE, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines()]

    valid_username = lines[0]
    valid_password = lines[1]

    if (username == valid_username) & (password == valid_password):
        return True
    else:
        return False


# Define elements of the UI
model_name = gr.Radio(loader.AUTHORIZED_MODELS, value=DEFAULT, label='Model name',
                      info='Choose the model you want to use.')
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt')
max_new_tokens = gr.Slider(10, 1000, value=100, step=5, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
do_sample = gr.Checkbox(value=True, label='Sampling', info='Whether to incorporate randomness in generation.')
top_k = gr.Slider(0, 200, value=100, step=5, label='Top-k',
               info='How many tokens with max probability to consider for randomness.')
top_p = gr.Slider(0, 1, value=0.92, step=0.01, label='Top-p',
              info='The probability density covering the new tokens to consider for randomness')
temperature = gr.Slider(0, 1, value=0.9, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')
num_return_sequence = gr.Slider(1, 10, value=1, step=1, label='Sequence', info='Number of sequence to generate.')
use_seed = gr.Checkbox(value=False, label='Use seed', info='Whether to use a fixed seed for reproducibility.')
seed = gr.Number(0, label='Seed', info='Seed for reproducibility.', precision=0)
output = gr.Textbox(label='Model output')
generate_button = gr.Button('Generate text', variant='primary')
clear_button = gr.Button('Clear prompt')
flag_button = gr.Button('Flag', variant='stop')

# Define the inputs for the main inference
inputs_to_main = [prompt, max_new_tokens, do_sample, top_k, top_p, temperature, num_return_sequence, use_seed, seed]
# set-up callback for flagging
callback = gr.CSVLogger()


demo = gr.Blocks(title='Text generation with LLMs')

with demo:
    model_name.render()
    with gr.Row():
        max_new_tokens.render()
        do_sample.render()
        top_k.render()
        top_p.render()
    with gr.Row():
        temperature.render()
        num_return_sequence.render()
        use_seed.render()
        seed.render()
    prompt.render()
    with gr.Row():
        generate_button.render()
        clear_button.render()
    output.render()
    flag_button.render()

    # Perform text generation when clicking the button or pressing enter in the prompt box
    generate_event1 = generate_button.click(text_generation, inputs=inputs_to_main, outputs=output)
    generate_event2 = prompt.submit(text_generation, inputs=inputs_to_main, outputs=output)

    # Switch the model loaded in memory when clicking on a new model
    model_name.input(update_model, inputs=model_name, cancels=[generate_event1, generate_event2])

    # Clear the prompt box when clicking the button
    clear_button.click(lambda: gr.update(value=''), outputs=prompt)

    # Perform the flagging
    callback.setup([model_name, *inputs_to_main, output], flagging_dir='flagged')
    flag_button.click(lambda *args: callback.flag(args), inputs=[model_name, *inputs_to_main, output], preprocess=False)


if __name__ == '__main__':
    demo.queue().launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])  
