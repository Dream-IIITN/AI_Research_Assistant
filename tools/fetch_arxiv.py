from langchain_core.tools import tool
import requests
import re

abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

@tool('fetch_arxiv')
def fetch_arxiv(arxiv_id: str) -> str:
    '''Fetches the abstract from an ArXiv paper given its ArXiv ID.'''
    try:
        res = requests.get(f'https://arxiv.org/abs/{arxiv_id}', timeout=10)
        res.raise_for_status()
        match = abstract_pattern.search(res.text)
        return match.group(1) if match else "Abstract not available."
    except requests.exceptions.RequestException as e:
        return f"Error fetching abstract: {str(e)}"
