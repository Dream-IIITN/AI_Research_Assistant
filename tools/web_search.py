from langchain_core.tools import tool
from serpapi import GoogleSearch
from utils.config import serpapi_params

@tool('web_search')
def web_search(query: str) -> str:
    '''Finds general knowledge information using a Google search.'''
    try:
        search = GoogleSearch({**serpapi_params, 'q': query, 'num': 5})
        results = search.get_dict().get('organic_results', [])
        if not results:
            return "No results found."
        formatted_results = '\n---\n'.join(
            [f"{x['title']}\n{x['snippet']}\n{x['link']}" for x in results]
        )
        return formatted_results
    except Exception as e:
        return f"Error during web search: {str(e)}"
