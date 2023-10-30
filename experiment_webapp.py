import os
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import gradio as gr

import engine
from helpers import utils
from engine.streamer import TextContinuationStreamer
from engine.conversation_template import GenericConversation


# Default model to load at start-up
DEFAULT = 'model1'

# Load both models at the beginning for fast switch between them later on
MODELS = [engine.HFModel('llama2-70B-chat'), engine.HFModel('llama2-13B-chat', gpu_rank=5)]

# Mapping to mask actual model name
MAPPING = {f'model{i+1}': MODELS[i].model_name for i in range(len(MODELS))}

MAPPING_TO_INDICES = {f'model{i+1}': i for i in range(len(MODELS))}

# Reverse mapping
REVERSE_MAPPING = {value: key for key, value in MAPPING.items()}

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# This will be a mapping between users and current models, to reload them with page reload
CACHED_MODELS = {}

# Need to define one logger per user
LOGGERS = {}


def update_model(new_masked_model_name: str, username: str) -> tuple[GenericConversation, str, str, str, list[list[str]]]:
    """Update the "true" model name and clear the conversation (even if we ask for the same model as the old
    one).

    Parameters
    ----------
    new_masked_model_name : str
        The masked name for the new model.
    username : str
        The username of the current session.

    Returns
    -------
    tuple[GenericConversation, str, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, model_name, prompt_chat, output_chat)
    """

    index = MAPPING_TO_INDICES[new_masked_model_name]
    model = MODELS[index]

    conversation = model.get_empty_conversation()
    # Update cached values
    CACHED_CONVERSATIONS[username] = conversation
    CACHED_MODELS[username] = model.model_name

    return conversation, conversation.id, model.model_name, '', conversation.to_gradio_format()
    


def chat_generation(model_name: str, conversation: GenericConversation, prompt: str,
                    max_new_tokens: int = 60) -> tuple[GenericConversation, str, list[list[str, str]]]:
    """Chat generation with streamed output.

    Parameters
    ----------
    model_name : str
        Current "true" model name.
    prompt : str
        Prompt to the model.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int, optional
        Maximum new tokens to generate, by default 60

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, prompt_chat, output_chat)
    """

    # Compute the reference to the model (already loaded) that we use
    masked_model_name = REVERSE_MAPPING[model_name]
    index = MAPPING_TO_INDICES[masked_model_name]
    model = MODELS[index]

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
                yield conversation, '', conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, '', conversation.to_gradio_format()



def continue_generation(model_name: str, conversation: GenericConversation,
                        additional_max_new_tokens: int = 60) -> tuple[GenericConversation, str, list[list[str, str]]]:
    """Continue the last turn of the conversation, with streamed output.

    Parameters
    ----------
    model_name : str
        Current "true" model name.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int, optional
        Maximum new tokens to generate, by default 60

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, prompt_chat, output_chat)
    """
   
   # Compute the reference to the model (already loaded) that we use
    masked_model_name = REVERSE_MAPPING[model_name]
    index = MAPPING_TO_INDICES[masked_model_name]
    model = MODELS[index]

    timeout = 20

    # To show text as it is being generated
    streamer = TextContinuationStreamer(model.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(model.continue_last_conversation_turn, conv_history=conversation,
                                 max_new_tokens=additional_max_new_tokens, do_sample=True, top_k=None, top_p=0.9,
                                 temperature=0.8, seed=None, truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = conv_copy.model_history_text[-1]
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conversation, '', conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, '', conversation.to_gradio_format()



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
            return True
    
    return False
    

def clear_chatbot(model_name: str, username: str) -> tuple[GenericConversation, str, str, list[list[str]]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    model_name : str
        Current "true" model name.
    username : str
        The username of the current session.

    Returns
    -------
    tuple[GenericConversation, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, prompt_chat, output_chat)
    """

    # Compute the reference to the model (already loaded) that we use
    masked_model_name = REVERSE_MAPPING[model_name]
    index = MAPPING_TO_INDICES[masked_model_name]
    model = MODELS[index]

    # Create new global conv object (we need a new unique id)
    conversation = model.get_empty_conversation()
    # Cache value
    CACHED_CONVERSATIONS[username] = conversation
    return conversation, conversation.id, '', conversation.to_gradio_format()



def loading(model_name: str, request: gr.Request) -> tuple[GenericConversation, str, str, str, str, list[list[str]]]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    model_name : str
        Current "true" model name.
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, str, str, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, username, masked_model_name, model_name, output_chat)
    """

    # Retrieve username
    if request is not None:
        username = request.username
    else:
        raise RuntimeError('Impossible to find username on startup.')
    
    # Check if we have cached values
    if username in CACHED_MODELS.keys():
        model_name = CACHED_MODELS[username]
    else:
        # default value of the element
        CACHED_MODELS[username] = model_name

    masked_model_name = REVERSE_MAPPING[model_name]
    index = MAPPING_TO_INDICES[masked_model_name]
    model = MODELS[index]
    
    # Check if we have cached a value for the conversation to use
    if username in CACHED_CONVERSATIONS.keys():
        actual_conv = CACHED_CONVERSATIONS[username]
    else:
        actual_conv = model.get_empty_conversation()
        CACHED_CONVERSATIONS[username] = actual_conv
        LOGGERS[username] = gr.CSVLogger()

    conv_id = actual_conv.id
    
    return actual_conv, conv_id, username, masked_model_name, model.model_name, actual_conv.to_gradio_format()
    



# Define generation parameters and model selection
masked_model_name = gr.Dropdown(list(MAPPING.keys()), value=DEFAULT, label='Model name',
                                info='Choose the model you want to use.', multiselect=False)
max_new_tokens = gr.Slider(10, 4000, value=500, step=10, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(1, 500, value=100, step=1, label='Max additional new tokens',
                           info='Maximum number of new tokens to generate when using "Continue last answer" feature.')
load_button = gr.Button('Load model', variant='primary')


# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('Generate text', variant='primary')
continue_button_chat = gr.Button('Continue last answer', variant='primary')
clear_button_chat = gr.Button('Clear conversation')


# State variable to keep one conversation per session (default value does not matter here -> it will be set
# by loading() method anyway)
conversation = gr.State(MODELS[0].get_empty_conversation())


# Define NON-VISIBLE elements: only used to save variables to the callback. Somewhat used as state variables.
model_name = gr.Dropdown(list(REVERSE_MAPPING.keys()), value=MAPPING[DEFAULT], label='Model name',
                         multiselect=False, visible=False)
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)


# Define the inputs for the main inference
inputs_to_chatbot = [prompt_chat, max_new_tokens]
# Define inputs for the logging callbacks
inputs_to_chat_callback = [model_name, max_new_tokens, max_additional_new_tokens, output_chat, conv_id, username]


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

    # State variable
    conversation.render()

    # State variables that do not need to be updated (except conv_id but the value is mostly fix and we know when
    # to update it) -- will not be visible
    model_name.render()
    conv_id.render()
    username.render()

    # Need to wrap everything in a row because we want two side-by-side columns
    with gr.Row():

        # First column where we have prompts and outputs. We use large scale because we want a 1.7:1 ratio
        # but scale needs to be an integer
        with gr.Column(scale=17):

            prompt_chat.render()
            with gr.Row():
                generate_button_chat.render()
                clear_button_chat.render()
                continue_button_chat.render()
            output_chat.render()

            gr.Markdown("### Prompt Examples")
            gr.Examples(prompt_examples, inputs=prompt_chat)

        # Second column defines model selection and generation parameters
        with gr.Column(scale=10):
                
            # First box for model selection
            with gr.Box():
                gr.Markdown("### Model selection")
                with gr.Row():
                    masked_model_name.render()
                with gr.Row():
                    load_button.render()
            
            # Accordion for generation parameters
            with gr.Accordion("Text generation parameters", open=False):
                max_new_tokens.render()
                max_additional_new_tokens.render()


    # Perform chat generation when clicking the button
    generate_event1 = generate_button_chat.click(chat_generation, inputs=[model_name, conversation, *inputs_to_chatbot],
                                                 outputs=[conversation, prompt_chat, output_chat])

    # Add automatic callback on success (args[-1] is the username)
    generate_event1.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'generation'),
                            inputs=inputs_to_chat_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button_chat.click(continue_generation, inputs=[model_name, conversation, max_additional_new_tokens],
                                                 outputs=[conversation, prompt_chat, output_chat])
    
    # Add automatic callback on success (args[-1] is the username)
    generate_event2.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'continuation'),
                            inputs=inputs_to_chat_callback, preprocess=False)

    # Switch the model loaded in memory when clicking
    events_to_cancel = [generate_event1, generate_event2]
    load_event = load_button.click(update_model, inputs=[masked_model_name, username],
                                   outputs=[conversation, conv_id, model_name, prompt_chat, output_chat],
                                   cancels=events_to_cancel)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_chat.click(clear_chatbot, inputs=[model_name, username], outputs=[conversation, conv_id, prompt_chat, output_chat])
    
    # Correctly set all variables and callback at load time
    loading_events = demo.load(loading, inputs=model_name,
                               outputs=[conversation, conv_id, username, masked_model_name, model_name, output_chat])
    loading_events.then(lambda username: LOGGERS[username].setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs/{username}'),
                        inputs=username)


if __name__ == '__main__':
    demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
