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


# Load model
MODEL = engine.HFModel('llama2-70B-chat')

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# Need to define one logger per user
LOGGERS = {}


def chat_generation(conversation: GenericConversation, prompt: str,
                    max_new_tokens: int) -> tuple[GenericConversation, str, list[list[str, str]]]:
    """Chat generation with streamed output.

    Parameters
    ----------
    prompt : str
        Prompt to the model.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int
        Maximum new tokens to generate.

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, prompt_chat, output_chat)
    """

    timeout = 20

    # To show text as it is being generated
    streamer = TextIteratorStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(MODEL.generate_conversation, prompt, system_prompt='', conv_history=conversation,
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



def continue_generation(conversation: GenericConversation,
                        additional_max_new_tokens) -> tuple[GenericConversation, str, list[list[str, str]]]:
    """Continue the last turn of the conversation, with streamed output.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int
        Maximum new tokens to generate.

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, prompt_chat, output_chat)
    """
   
    timeout = 20

    # To show text as it is being generated
    streamer = TextContinuationStreamer(MODEL.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(MODEL.continue_last_conversation_turn, conv_history=conversation,
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
    

def clear_chatbot(username: str) -> tuple[GenericConversation, str, str, list[list[str]]]:
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

    # Create new global conv object (we need a new unique id)
    conversation = MODEL.get_empty_conversation()
    # Cache value
    CACHED_CONVERSATIONS[username] = conversation

    return conversation, conversation.id, '', conversation.to_gradio_format()



def loading(request: gr.Request) -> tuple[GenericConversation, str, str, str, str, list[list[str]]]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, str, str, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, username, model_name, output_chat)
    """

    # Retrieve username
    if request is not None:
        username = request.username
    else:
        raise RuntimeError('Impossible to find username on startup.')
    
    # Check if we have cached a value for the conversation to use
    if username in CACHED_CONVERSATIONS.keys():
        actual_conv = CACHED_CONVERSATIONS[username]
    else:
        actual_conv = MODEL.get_empty_conversation()
        CACHED_CONVERSATIONS[username] = actual_conv
        LOGGERS[username] = gr.CSVLogger()

    conv_id = actual_conv.id
    
    return actual_conv, conv_id, username, MODEL.model_name, actual_conv.to_gradio_format()
    



# Define generation parameters
max_new_tokens = gr.Slider(10, 4000, value=500, step=10, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(1, 500, value=100, step=1, label='Max additional new tokens',
                           info='Maximum number of new tokens to generate when using "Continue last answer" feature.')


# Define elements of the chatbot Tab
prompt_chat = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output_chat = gr.Chatbot(label='Conversation')
generate_button_chat = gr.Button('Generate text', variant='primary')
continue_button_chat = gr.Button('Continue last answer', variant='primary')
clear_button_chat = gr.Button('Clear conversation')


# State variable to keep one conversation per session (default value does not matter here -> it will be set
# by loading() method anyway)
conversation = gr.State(MODEL.get_empty_conversation())


# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback.
model_name = gr.Textbox(MODEL.model_name, label='Model Name', visible=False)
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

    # Variables we track with usual components: they do not need to be State variables -- will not be visible
    model_name.render()
    conv_id.render()
    username.render()

    # Actual UI
    output_chat.render()
    prompt_chat.render()
    with gr.Row():
        generate_button_chat.render()
        continue_button_chat.render()
        clear_button_chat.render()

    # Accordion for generation parameters
    with gr.Accordion("Text generation parameters", open=False):
        max_new_tokens.render()
        max_additional_new_tokens.render()

    gr.Markdown("### Prompt Examples")
    gr.Examples(prompt_examples, inputs=prompt_chat)


    # Perform chat generation when clicking the button
    generate_event1 = generate_button_chat.click(chat_generation, inputs=[conversation, *inputs_to_chatbot],
                                                 outputs=[conversation, prompt_chat, output_chat])

    # Add automatic callback on success (args[-1] is the username)
    generate_event1.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'generation'),
                            inputs=inputs_to_chat_callback, preprocess=False, queue=False)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button_chat.click(continue_generation, inputs=[conversation, max_additional_new_tokens],
                                                 outputs=[conversation, prompt_chat, output_chat])
    
    # Add automatic callback on success (args[-1] is the username)
    generate_event2.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'continuation'),
                            inputs=inputs_to_chat_callback, preprocess=False, queue=False)
    
    # Clear the prompt and output boxes when clicking the button
    clear_button_chat.click(clear_chatbot, inputs=[username], outputs=[conversation, conv_id, prompt_chat, output_chat],
                            queue=False)
    
    # Correctly set all variables and callback at load time
    loading_events = demo.load(loading, outputs=[conversation, conv_id, username, model_name, output_chat], queue=False)
    loading_events.then(lambda username: LOGGERS[username].setup(inputs_to_chat_callback, flagging_dir=f'chatbot_logs/{username}'),
                        inputs=username, queue=False)


if __name__ == '__main__':
    demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
