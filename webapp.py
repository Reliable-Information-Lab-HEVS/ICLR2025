import gradio as gr
import gc
import os

import loader
import engine
import utils


# Default model to load at start-up
DEFAULT = 'bloom-560M'

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# Initialize global model and tokenizer (necessary not to reload the model for each new inference)
model, tokenizer = loader.load_model_and_tokenizer(DEFAULT)

# Initialize a global conversation object for chatting with the models
conversation = engine.Conversation()


def update_model(model_name: str, quantization:bool = False):
    """Update the model and tokenizer in the global scope so that we can reuse it and speed up inference.

    Parameters
    ----------
    model_name : str
        The name of the new model to use.
    quantization : bool, optional
        Whether to load the model in 8 bits mode.
    """
    
    global model
    global tokenizer

    # Delete the variables if they exist (they should except if there was an error when loading a model at some point)
    try:
        del model
        del tokenizer
    except NameError:
        pass
    gc.collect()

    # Try loading the model and tokenizer
    try:
        model = loader.load_model(model_name, quantization)
        tokenizer = loader.load_tokenizer(model_name)
    except:
        gr.Error('There was an error loading this model. Please choose another one.')

    # Clear current conversation (if any) when switching the model
    conversation.erase_conversation()


def text_generation(prompt: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 40,
                    top_p: float = 0.90, temperature: float = 0.9, num_return_sequences: int = 1,
                    use_seed: bool = False, seed: int | None = None) -> str:
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
        Whether to use a fixed seed for reproducibility, by default False.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.
    Returns
    -------
    str
        String containing all `num_return_sequences` sequences generated.
    """
    
    if not use_seed:
        seed = None
    predictions = engine.generate_text(model, tokenizer, prompt, max_new_tokens, do_sample, top_k, top_p,
                                       temperature, num_return_sequences, seed)
    return utils.format_output(predictions)



def chat_generation(prompt: str, max_new_tokens: int = 60, do_sample: bool = True, top_k: int = 40, top_p: float = 0.90,
                    temperature: float = 0.9, use_seed: bool = False, seed: int | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Chat generation using the model, tokenizer and conversation in the global scope, so that we can reuse them for multiple
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
    use_seed : bool, optional
        Whether to use a fixed seed for reproducibility., by default False.
    seed : Union[None, int], optional
        An optional seed to force the generation to be reproducible.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        An empty string to reinitialize the prompt box, and the conversation as a list[tuple[str, str]].
    """
    
    if not use_seed:
        seed = None
    
    # This will update the global conversation in-place
    _ = engine.generate_conversation(model, tokenizer, prompt, conversation, max_new_tokens, do_sample, top_k,
                                     top_p, temperature, seed)
    # The first output is an empty string to clear the input box, the second cast the 2 list of chat history into a single list of tuples (user, model)
    return '', list(zip(conversation.user_history_text, conversation.model_history_text))



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
    

def clear_chatbot():
    """Erase the conversation history and reinitialize the elements.
    """

    # Erase the conversation history before
    conversation.erase_conversation()
    return '', [(None, None)]
    


# Define general elements of the UI (generation parameters)
model_name = gr.Dropdown(loader.AUTHORIZED_MODELS, value=DEFAULT, label='Model name',
                         info='Choose the model you want to use.', multiselect=False)
quantization = gr.Checkbox(value=True, label='Quantization', info='Whether to load the model in 8 bits mode.')
max_new_tokens = gr.Slider(10, 200, value=50, step=5, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
do_sample = gr.Checkbox(value=True, label='Sampling', info='Whether to incorporate randomness in generation.')
top_k = gr.Slider(0, 200, value=40, step=5, label='Top-k',
               info='How many tokens with max probability to consider.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens.')
temperature = gr.Slider(0, 1, value=0.9, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')
num_return_sequence = gr.Slider(1, 10, value=1, step=1, label='Sequence', info='Number of sequence to generate.')
use_seed = gr.Checkbox(value=False, label='Use seed', info='Whether to use a fixed seed for reproducibility.')
seed = gr.Number(0, label='Seed', info='Seed for reproducibility.', precision=0)

# Define elements of the simple generation Tab
prompt_text = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=10)
output_text = gr.Textbox(label='Model output')
generate_button_text = gr.Button('Generate text', variant='primary')
clear_button_text = gr.Button('Clear')
flag_button_text = gr.Button('Flag', variant='stop')

# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=10)
output_chat = gr.Chatbot()
generate_button_chat = gr.Button('Generate text', variant='primary')
clear_button_chat = gr.Button('Clear')
flag_button_chat = gr.Button('Flag', variant='stop')

# Define the inputs for the main inference
inputs_to_simple_generation = [prompt_text, max_new_tokens, do_sample, top_k, top_p, temperature, num_return_sequence, use_seed, seed]
inputs_to_chatbot = [prompt_chat, max_new_tokens, do_sample, top_k, top_p, temperature, use_seed, seed]
# set-up callback for flagging
callback = gr.CSVLogger()


demo = gr.Blocks(title='Text generation with LLMs')

with demo:

    # Need to wrap everything in a row because we want two side-by-side columns
    with gr.Row():

        # First column where we have prompts and outputs
        with gr.Column(scale=2):

            # Tab 1 for simple text generation
            with gr.Tab('Text generation'):
                prompt_text.render()
                with gr.Row():
                    generate_button_text.render()
                    clear_button_text.render()
                output_text.render()
                flag_button_text.render()

            # Tab 2 for chat mode
            with gr.Tab('Chat mode'):
                prompt_chat.render()
                with gr.Row():
                    generate_button_chat.render()
                    clear_button_chat.render()
                output_chat.render()

        # Second column defines model selection and generation parameters
        with gr.Column(scale=1):
                
                # First box for model selection
                with gr.Box():
                    gr.Markdown("### Model selection")
                    with gr.Row():
                        model_name.render()
                        quantization.render()
                
                # Second box for generation parameters
                with gr.Box():
                    gr.Markdown("### Text generation parameters")
                    with gr.Row():
                        max_new_tokens.render()
                        do_sample.render()
                    with gr.Row():
                        top_k.render()
                        top_p.render()
                    with gr.Row():
                        temperature.render()
                        num_return_sequence.render()
                    with gr.Row():
                        use_seed.render()
                        seed.render()

    # Perform simple text generation when clicking the button or pressing enter in the prompt box
    generate_event1 = generate_button_text.click(text_generation, inputs=inputs_to_simple_generation, outputs=output_text)
    generate_event2 = prompt_text.submit(text_generation, inputs=inputs_to_simple_generation, outputs=output_text)

    # Perform chat generation when clicking the button or pressing enter in the prompt box
    generate_event3 = generate_button_chat.click(chat_generation, inputs=inputs_to_chatbot, outputs=[prompt_chat, output_chat])
    generate_event4 = prompt_chat.submit(chat_generation, inputs=inputs_to_chatbot, outputs=[prompt_chat, output_chat])

    # Switch the model loaded in memory when clicking on a new model or changing quantization and clear outputs if any
    events_to_cancel = [generate_event1, generate_event2, generate_event3, generate_event4]
    model_name.input(update_model, inputs=[model_name, quantization], cancels=events_to_cancel).then(lambda: '', outputs=output_text).then(clear_chatbot, outputs=[prompt_chat, output_chat])
    quantization.input(update_model, inputs=[model_name, quantization], cancels=events_to_cancel).then(lambda: '', outputs=output_text).then(clear_chatbot, outputs=[prompt_chat, output_chat])
    # load_button.click(update_model, inputs=[model_name, quantization], cancels=[generate_event1, generate_event2])
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_text.click(lambda: ['', ''], outputs=[prompt_text, output_text])
    clear_button_chat.click(clear_chatbot, outputs=[prompt_chat, output_chat])

    # Perform the flagging
    callback.setup([model_name, *inputs_to_simple_generation, output_text], flagging_dir='flagged')
    flag_button_text.click(lambda *args: callback.flag(args), inputs=[model_name, *inputs_to_simple_generation, output_text], preprocess=False)



if __name__ == '__main__':
    demo.queue().launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
