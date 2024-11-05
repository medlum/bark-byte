from streamlit_chat import message
import streamlit as st
from utils_agent_tools import *
from utils_prompt import *
from utils_tts import *
from streamlit_extras.bottom_container import bottom
import streamlit_antd_components as sac
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from huggingface_hub.errors import OverloadedError
import re
from streamlit_extras.grid import grid
# ---------set up page config -------------#
st.set_page_config(page_title="Bark & Byte",
                   layout="wide", page_icon="🐶")

# ---------Inject CSS for buttons -------------#

st.markdown(custom_css, unsafe_allow_html=True)

# ---- set up creative chat history ----#
chat_msg = StreamlitChatMessageHistory(key="chat_key")
chat_history_size = 5

# ---------set up LLM  -------------#
# model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model = "Qwen/Qwen2.5-72B-Instruct"

# initialise LLM for agents and tools
llm_factual = HuggingFaceEndpoint(
    repo_id=model,
    max_new_tokens=1000,
    do_sample=False,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# ---------set up general memory  -------------#
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msg,
    k=chat_history_size,
    return_messages=True
)

# ---------set up agent with tools  -------------#
react_agent = create_react_agent(llm_factual, toolkit, prompt)

executor = AgentExecutor(
    agent=react_agent,
    tools=toolkit,
    memory=conversational_memory,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs=agent_kwargs,
)


# ------ initial welcome message -------#

# set up session state as a gate to display welcome message
if 'initial_msg' not in st.session_state:
    st.session_state.initial_msg = 0

# if 0, add welcome message to chat_msg
if st.session_state.initial_msg == 0:
    part_day = get_time_bucket()  # located at utils_tts.py
    welcome_msg = f"{part_day} How can I help you today?"
    chat_msg.add_ai_message(welcome_msg)

# "personas"
avatar_style = "personas"
seed_user = "Ryker"
seed_bot = "Angel"
# ------ set up message from chat history  -----#

for index, msg in enumerate(chat_msg.messages):

    # bot's message is in even position as welcome message is added at initial
    if index % 2 == 0:

        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("AI:", "").replace("Human:", "").replace("<|endoftext|>", ""),
                is_user=False,
                key=f"bot{index}",
                avatar_style=avatar_style,
                seed=seed_bot,
                allow_html=True,
                is_table=True,)

    # user's message is in odd position
    else:

        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", ""),
                is_user=True,
                key=f"user{index}",
                avatar_style=avatar_style,
                seed=seed_user)

# set initial_msg to 0 in first loop
if index == 0:
    st.session_state.initial_msg = 1


button_pressed = ""
# use streamlit_extras to create grids
btn_grid = grid(3, vertical_align="top")

if btn_grid.button(example_prompts[0]):
    button_pressed = example_prompts[0]
elif btn_grid.button(example_prompts[1]):
    button_pressed = example_prompts[1]
elif btn_grid.button(example_prompts[2]):
    button_pressed = example_prompts[2]
elif btn_grid.button(example_prompts[3]):
    button_pressed = example_prompts[3]
elif btn_grid.button(example_prompts[4]):
    button_pressed = example_prompts[4]
elif btn_grid.button(example_prompts[5]):
    button_pressed = example_prompts[5]
elif btn_grid.button(example_prompts[6]):
    button_pressed = example_prompts[6]

# ------ set up user input -----#

if prompt := (st.chat_input("Ask me a question...") or button_pressed):

    # show prompt message
    message(prompt,
            is_user=True,
            key=f"user",
            avatar_style=avatar_style,
            seed=seed_user)

# ---- if response_type is agent -----#

    with st.spinner("Generating..."):

        response = executor.invoke(
            {'input': f'<|im_start|>{prompt}<|im_end|>'})

        # remove prompt format for better display
        edited_response = str(
            response['output'].replace('<|eot_id|>', ''))

        # show message
        message(edited_response,
                is_user=False,
                key=f"bot_1",
                avatar_style=avatar_style,
                seed=seed_bot,
                allow_html=True,
                is_table=True)

        st.rerun()
