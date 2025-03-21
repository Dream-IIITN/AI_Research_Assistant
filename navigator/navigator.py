from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from getpass import getpass
import os

system_prompt = (
    '''You are the Navigator, the great AI decision-maker.
    Given the user's query, you must decide what to do with it based on the
    list of tools provided to you.

    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice.

    You should aim to collect information from a diverse range of sources before
    providing the answer to the user. Once you have collected plenty of information
    (stored in the scratchpad), use the final_answer tool.
    '''
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
    ('assistant', 'scratchpad: {scratchpad}'),
])

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    groq_api_key=os.getenv('GROQ_API_KEY') or getpass('YourAPI key: '),
    temperature=0
)

# Tools will be bound later in the decision pipeline.
