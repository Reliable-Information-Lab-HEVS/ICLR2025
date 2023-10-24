import gc
import os
import time
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import torch
import gradio as gr

import engine
from engine import loader
from helpers import utils


# Default model to load at start-up
DEFAULT = 'model1'

# Mapping to mask actual model name
MAPPING = {
    'model1': 'llama2-70B-chat',
    'model2': 'llama2-13B-chat',
}

# Reverse mapping
REVERSE_MAPPING = {value: key for key, value in MAPPING.items()}

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be setup by the authentication method
USERNAME = None

# Initialize global model (necessary not to reload the model for each new inference)
model = engine.HFModel(MAPPING[DEFAULT])

# TODO: make conversation a session state variable instead of global state variable
# Initialize a global conversation object for chatting with the models
conversation = model.get_empty_conversation()


def update_model(masked_model_name: str):
    """Update the model and conversation in the global scope so that we can reuse them and speed up inference.

    Parameters
    ----------
    masked_model_name : str
        The anonymous name of the new model to use.
    quantization : bool, optional
        Whether to load the model in 8 bits mode.
    """

    model_name = MAPPING[masked_model_name]

    global model
    global conversation

    try:
        # If we ask for the same setup, do nothing
        if model_name == model.model_name:
            return '', '', '', [[None, None]]
    except NameError:
        pass

    # Delete the variables if they exist (they should except if there was an error when loading a model at some point)
    # to save memory before loading the new one
    try:
        del model
        del conversation
        gc.collect()
    except NameError:
        pass

    # Try loading the model
    try:
        model = engine.HFModel(model_name, quantization_8bits=False, quantization_4bits=False)
        conversation = model.get_empty_conversation()
    except Exception as e:
        raise gr.Error(f'The following error happened during loading: {repr(e)}. Please retry or choose another one.')
    
    # Return values to clear the input and output textboxes, and input and output chatbot boxes
    # return '', '', '', [[None, None]]
    return '', [[None, None]]


def text_generation(prompt: str, max_new_tokens: int = 60) -> str:
    """Text generation using the model and tokenizer in the global scope, so that we can reuse them for multiple
    prompts.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.

    Returns
    -------
    str
        String containing the sequence generated.
    """

    timeout = 10

    # To show text as it is being generated
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.generate_text, prompt, max_new_tokens=max_new_tokens, do_sample=True,
                                 top_k=None, top_p=0.9, temperature=0.8, seed=None,
                                 truncate_prompt_from_output=True, streamer=streamer)
    
        # Get results from the streamer and yield it
        try:
            # Ask the streamer to skip prompt and reattach it here to avoid showing special prompt formatting
            generated_text = prompt
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
        # Get actual result and yield it (which may be slightly different due to postprocessing)
        generated_text = future.result()
        yield prompt + generated_text


def chat_generation(prompt: str, max_new_tokens: int = 60) -> tuple[str, list[tuple[str, str]]]:
    """Chat generation using the model, tokenizer and conversation in the global scope, so that we can reuse
    them for multiple prompts.

    Parameters
    ----------
    prompt : str
        The prompt to the model.
    max_new_tokens : int, optional
        How many new tokens to generate, by default 60.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        An empty string to reinitialize the prompt box, and the conversation as a list[tuple[str, str]].
    """

    timeout = 20

    # To show text as it is being generated
    streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(model.generate_conversation, prompt, system_prompt='', conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=True, top_k=None, top_p=0.9,
                                 temperature=0.8, seed=None, truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield '', conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield '', conversation.to_gradio_format()


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
        Return True if both the username and password match some credentials stored in `CREDENTIALS_FILE`. 
        False otherwise.
    """

    with open(CREDENTIALS_FILE, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines() if line.strip() != '']

    valid_usernames = lines[0::2]
    valid_passwords = lines[1::2]

    if username in valid_usernames:
        index = valid_usernames.index(username)
        # Check that the password also matches at the corresponding index
        if password == valid_passwords[index]:
            # Save the username in a global variable for later access
            global USERNAME
            USERNAME = username
            return True
    
    return False
    

def clear_chatbot():
    """Erase the conversation history and reinitialize the elements.
    """

    # Erase the conversation history before
    conversation.erase_conversation()
    return '', conversation.to_gradio_format()
    


# Define general elements of the UI (generation parameters)
model_name = gr.Dropdown(list(MAPPING.keys()), value=DEFAULT, label='Model name',
                         info='Choose the model you want to use.', multiselect=False)
max_new_tokens = gr.Slider(10, 4000, value=500, step=10, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
load_button = gr.Button('Load model', variant='primary')

# Define elements of the simple generation Tab
# prompt_text = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
# output_text = gr.Textbox(label='Model output')
# generate_button_text = gr.Button('Generate text', variant='primary')
# clear_button_text = gr.Button('Clear prompt', variant='secondary')

# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('Generate text', variant='primary')
clear_button_chat = gr.Button('Clear conversation')

# Define the inputs for the main inference
# inputs_to_simple_generation = [prompt_text, max_new_tokens]
inputs_to_chatbot = [prompt_chat, max_new_tokens]

# Define inputs for the logging callbacks
# inputs_to_text_callback = [model_name, *inputs_to_simple_generation,
#                            output_text]
inputs_to_chat_callback = [model_name, *inputs_to_chatbot,
                           output_chat]

# set-up callbacks for flagging and automatic logging
# automatic_logging_text = gr.CSVLogger()
automatic_logging_chat = gr.CSVLogger()

# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='Text generation with LLMs')

with demo:

    # Need to wrap everything in a row because we want two side-by-side columns
    with gr.Row():

        # First column where we have prompts and outputs. We use large scale because we want a 1.7:1 ratio
        # but scale needs to be an integer
        with gr.Column(scale=17):

            # Tab 1 for simple text generation
            # with gr.Tab('Text generation'):
            #     prompt_text.render()
            #     with gr.Row():
            #         generate_button_text.render()
            #         clear_button_text.render()
            #     output_text.render()

            #     gr.Markdown("### Prompt Examples")
            #     gr.Examples(prompt_examples, inputs=prompt_text)

            # Tab 2 for chat mode
            # with gr.Tab('Chat mode'):
            prompt_chat.render()
            with gr.Row():
                generate_button_chat.render()
                clear_button_chat.render()
            output_chat.render()

            gr.Markdown("### Prompt Examples")
            gr.Examples(prompt_examples, inputs=prompt_chat)

        # Second column defines model selection and generation parameters
        with gr.Column(scale=10):
                
            # First box for model selection
            with gr.Box():
                gr.Markdown("### Model selection")
                with gr.Row():
                    model_name.render()
                with gr.Row():
                    load_button.render()
            
            # Accordion for generation parameters
            with gr.Accordion("Text generation parameters", open=False):
                with gr.Row():
                    max_new_tokens.render()

    # Perform simple text generation when clicking the button
    # generate_event1 = generate_button_text.click(text_generation, inputs=inputs_to_simple_generation,
                                                #  outputs=output_text)
    # Add automatic callback on success
    # generate_event1.success(lambda *args: automatic_logging_text.flag(args), inputs=inputs_to_text_callback,
                            # preprocess=False)

    # Perform chat generation when clicking the button
    generate_event2 = generate_button_chat.click(chat_generation, inputs=inputs_to_chatbot,
                                                 outputs=[prompt_chat, output_chat])

    # Add automatic callback on success
    generate_event2.success(lambda *args: automatic_logging_chat.flag(args), inputs=inputs_to_chat_callback,
                            preprocess=False)

    # Switch the model loaded in memory when clicking
    # events_to_cancel = [generate_event1, generate_event2]
    events_to_cancel = [generate_event2]
    # load_button.click(update_model, inputs=model_name,
                    #   outputs=[prompt_text, output_text, prompt_chat, output_chat], cancels=events_to_cancel)
    load_button.click(update_model, inputs=model_name,
                      outputs=[prompt_chat, output_chat], cancels=events_to_cancel)
    
    # Clear the prompt and output boxes when clicking the button
    # clear_button_text.click(lambda: ['', ''], outputs=[prompt_text, output_text])
    clear_button_chat.click(clear_chatbot, outputs=[prompt_chat, output_chat])
    
    # Correctly display the model and quantization currently on memory if we refresh the page (instead of default
    # value for the elements) and correctly reset the chat output
    loading_events = demo.load(lambda: [gr.update(value=REVERSE_MAPPING[model.model_name]),
                                        gr.update(value=conversation.to_gradio_format())],
                                outputs=[model_name, output_chat])
    
    # Set-up the flagging callbacks with updated USERNAME at loading time (in case of user change in the same session)
    # loading_events.then(lambda: [automatic_logging_text.setup(inputs_to_text_callback, flagging_dir=f'text_logs_{USERNAME}'),
                                #  automatic_logging_chat.setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs_{USERNAME}')])
    loading_events.then(lambda: automatic_logging_chat.setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs_{USERNAME}'))


if __name__ == '__main__':
    demo.queue().launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
