from langchain_core.tools import tool
from utils.chromadb_handler import chroma_db, embedding_function

def format_rag_contexts(query_result: dict) -> str:
    """
    Formats the ChromaDB query results into a readable string format.

    Args:
        query_result (dict): Dictionary containing 'ids', 'metadatas', and 'documents' from ChromaDB query.
    
    Returns:
        str: A formatted string of document titles, chunk excerpts, and paper IDs.
    """
    ids = query_result.get("ids", [])
    metadatas = query_result.get("metadatas", [])
    documents = query_result.get("documents", [])
    
    if not ids or len(ids) == 0:
        return "No relevant documents found."
    
    formatted_results = []
    for i in range(len(ids)):
        text = (
            f"Title: {metadatas[i].get('title', 'Unknown')}\n"
            f"Chunk: {documents[i][:200]}...\n"
            f"ArXiv ID: {metadatas[i].get('paper_id', 'N/A')}\n"
        )
        formatted_results.append(text)
    
    return "\n---\n".join(formatted_results)

@tool('rag_search_filter')
def rag_search_filter(query: str, arxiv_id: str) -> str:
    """
    Retrieves relevant research chunks from the ChromaDB chunks_collection for the given query,
    filtering by a specific ArXiv ID.
    
    Args:
        query (str): The natural language search query.
        arxiv_id (str): The ArXiv ID to filter the results by.
    
    Returns:
        str: Formatted string of matching document chunks.
    """
    xq = embedding_function([query])[0]
    results = chroma_db.chunks_collection.query(
        query_embeddings=[xq],
        n_results=6,
        where={"paper_id": arxiv_id}
    )
    return format_rag_contexts(results)

@tool('rag_search')
def rag_search(query: str) -> str:
    """
    Retrieves relevant research chunks from the ChromaDB chunks_collection based on the query.
    
    Args:
        query (str): The natural language search query.
    
    Returns:
        str: Formatted string of matching document chunks.
    """
    xq = embedding_function([query])[0]
    results = chroma_db.chunks_collection.query(
        query_embeddings=[xq],
        n_results=5
    )
    return format_rag_contexts(results)
