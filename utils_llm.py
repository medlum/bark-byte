from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils_agent_tools import *
from utils_prompt import *
from langchain.chains import LLMChain
from streamlit_app import chat_history_size, chat_msg
# ---------set up LLM  -------------#
llama3p1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# initialise LLM for agents and tools
llm_factual = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    max_new_tokens=1500,
    do_sample=False,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# ---- set up creative chat history ----#
# chat_msg = StreamlitChatMessageHistory(key="chat_key")
# chat_history_size = 5


# ---------set up general memory  -------------#
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msg,
    k=chat_history_size,
    return_messages=True)

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

# ---------set up for creative mode  -------------#
# Initialize LLM for creative mode
llm_creative = HuggingFaceEndpoint(
    repo_id=llama3p1_70B,
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# ------ set up the llm chain -----#
chat_llm_chain = LLMChain(
    llm=llm_creative,
    prompt=chatPrompt,  # located at utils_prompt.py
    verbose=True,
    memory=conversational_memory,
)
