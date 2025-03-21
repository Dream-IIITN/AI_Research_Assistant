#from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
# Load a sentence-transformer model
#model = SentenceTransformer("all-MiniLM-L6-v2")
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()

def encoder(texts: list) -> list:
    '''Encodes a list of texts into vector embeddings.'''
    return default_ef.encode(texts, convert_to_numpy=True)
