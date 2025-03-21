from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction
from navigator.navigator import prompt, llm
from tools.rag_search import rag_search, rag_search_filter
from tools.fetch_arxiv import fetch_arxiv
from tools.web_search import web_search
from tools.final_answer_tool import final_answer
from langgraph.graph import StateGraph, END
import operator
from typing import TypedDict, List, Annotated

# Define the state type
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# Helper function to create a scratchpad from intermediate steps
def create_scratchpad(intermediate_steps: List[tuple[AgentAction, str]]) -> str:
    research_steps = []
    for action, log_output in intermediate_steps:  
        if log_output != 'TBD':  
            research_steps.append(
                f"Tool: {action.tool}, Input: {action.tool_input}\nOutput: {log_output}"
            )
    return '\n---\n'.join(research_steps)

# Bind tools
tools = [rag_search, rag_search_filter, fetch_arxiv, web_search, final_answer]
navigator = (
    {
        'input': lambda x: x['input'],
        'chat_history': lambda x: x['chat_history'],
        'scratchpad': lambda x: create_scratchpad(intermediate_steps=x['intermediate_steps']),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice='any')
)

def run_navigator(state: dict) -> dict:
    '''Runs the navigator and processes the output to extract tool information.'''
    print('run_navigator')
    print(f"Intermediate steps: {state.get('intermediate_steps')}")

    out = navigator.invoke(state)

    # Ensure tool_calls exist
    if not out.tool_calls:
        print("ERROR: No tool_calls returned by navigator!")
        return state  

    # Extract tool name and arguments correctly
    tool_name = out.tool_calls[0]['name']  # ✅ Use dictionary access
    tool_args = out.tool_calls[0]['args']

    # ✅ Properly initialize AgentAction
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")

    # Append to `intermediate_steps` correctly
    return {
        'intermediate_steps': state.get('intermediate_steps', []) + [(action_out, "")]
    }

def router(state: dict) -> str:
    '''Determines the next tool to use based on the current state.'''
    if isinstance(state.get('intermediate_steps'), list) and state['intermediate_steps']:
        last_action = state['intermediate_steps'][-1][0]  
        return last_action.tool
    else:
        print('Router invalid format')
        return 'final_answer'

tool_str_to_func = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_arxiv': fetch_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}

def run_tool(state: dict) -> dict:
    '''Executes the appropriate tool based on the current state.'''
    last_action = state['intermediate_steps'][-1][0]  

    tool_name = last_action.tool
    tool_args = last_action.tool_input
    print(f"Running tool {tool_name} with input: {tool_args}")

    out = tool_str_to_func[tool_name].invoke(**tool_args)  # ✅ Unpack dictionary

    # ✅ Properly initialize AgentAction
    action_out = (AgentAction(tool=tool_name, tool_input=tool_args, log="Tool executed"), str(out))

    print("runned tool lol")  
    return {
        'intermediate_steps': state.get('intermediate_steps', []) + [action_out]
    }

def build_report(output: dict) -> str:
    '''Builds a formatted report based on the navigator's output.'''
    research_steps = output.get('research_steps', "")
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    
    sources = output.get('sources', "")
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    
    return f"""
INTRODUCTION
------------
{output.get('introduction', '')}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output.get('main_body', '')}

CONCLUSION
----------
{output.get('conclusion', '')}

SOURCES
-------
{sources}
"""

# Build the state graph
graph = StateGraph(AgentState)
graph.add_node('navigator', run_navigator)
graph.add_node('rag_search_filter', run_tool)
graph.add_node('rag_search', run_tool)
graph.add_node('fetch_arxiv', run_tool)
graph.add_node('web_search', run_tool)
graph.add_node('final_answer', run_tool)

graph.set_entry_point('navigator')
graph.add_conditional_edges(source='navigator', path=router)

# ✅ Fix `tool_obj.name`
for tool_obj in tools:
    if hasattr(tool_obj, 'name') and tool_obj.name != 'final_answer':  
        graph.add_edge(tool_obj.name, 'navigator')

graph.add_edge('final_answer', END)
runnable = graph.compile()
