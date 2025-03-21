from langchain_core.tools import tool

@tool('final_answer')
def final_answer(
    introduction: str,
    research_steps: str or list,
    main_body: str,
    conclusion: str,
    sources: str or list
) -> str:
    '''Returns a formatted research report string.'''
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    
    return (
        f"{introduction}\n\n"
        f"Research Steps:\n{research_steps}\n\n"
        f"Main Body:\n{main_body}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Sources:\n{sources}"
    )
