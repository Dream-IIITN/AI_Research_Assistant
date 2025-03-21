import chromadb
from chromadb.utils import embedding_functions 
import pandas as pd
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Initialize embedding function using SentenceTransformer
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Load CSV dataset containing research papers (assumed to be in the project root)
df = pd.read_csv('final_data.csv')
print("dataset loaded successfully!")
class ChromaDBHandler:
    def __init__(self, persist_directory="./chroma_db"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or load the collection for full research papers
        self.paper_collection = self.client.get_or_create_collection(
            name="research_papers",
            embedding_function=embedding_function
        )

        # Create or load the collection for paper chunks
        self.chunks_collection = self.client.get_or_create_collection(
            name="research_chunks",
            embedding_function=embedding_function
        )

    def download_pdf(self, pdf_url: str) -> str:
        """Downloads a PDF from the given URL and returns the local file path."""
        save_path = f"{pdf_url.split('/')[-1]}.pdf"
        try:
            response = requests.get(pdf_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
                print(f"✅ PDF downloaded successfully: {save_path}")
                return save_path
            else:
                print("❌ Failed to download PDF")
                return ""
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            return ""

    def store_paper(self):
        """
        Stores details of each research paper from the CSV into the ChromaDB paper_collection.
        """
        for _, r in df.iterrows():
            paper_id = str(r['entry_id'])
            metadata = {
                "title": r['title'],
                "pdf_url": r['pdf_url'],
                "authors": r['authors'],
                "published_year": r['published_year']
            }
            text_to_embed = f"{r['title']} {r['summary']}"
            embedding = embedding_function([text_to_embed])[0]  # Generate embedding

            self.paper_collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        print("✅ All papers stored in ChromaDB paper_collection.")

    def store_chunks(self, paper_id: str, pdf_path: str):
        """
        Downloads and splits a research paper PDF into chunks and stores them in the ChromaDB chunks_collection.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = text_splitter.split_documents(documents)
        
        # Store each chunk in the collection
        for i, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{i}"
            embedding = embedding_function([chunk.page_content])[0]
            metadata = {"paper_id": paper_id, "chunk_index": i}
            self.chunks_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[chunk.page_content]
            )
        print(f"✅ Stored {len(chunks)} chunks for paper {paper_id}.")

    def retrieve_relevant_papers(self, query_text: str, top_k: int = 5) -> list:
        """
        Retrieves the most relevant full papers from the ChromaDB paper_collection based on the query.
        """
        self.store_paper()
        query_embedding = embedding_function([query_text])[0]
        results = self.paper_collection.query(query_embeddings=[query_embedding], n_results=top_k)
        if not results.get('ids'):
            return []
        relevant_papers = []
        for i in range(len(results['ids'][0])):
            paper_info = {
                'id': results['ids'][0][i],
                'pdf_url': results['metadatas'][0][i].get('pdf_url', ''),
                'title': results['metadatas'][0][i].get('title', '')
            }
            relevant_papers.append(paper_info)
        return relevant_papers

    def retrieve_research_chunks(self, query_text: str, top_k: int = 5, iterations: int = 1) -> list:
        """
        Retrieves the most relevant chunks from the ChromaDB chunks_collection based on the query.
        """
        
        query_embedding = embedding_function([query_text])[0]
        all_chunks = []
        retrieved_ids = set()
        for _ in range(iterations):
            results = self.chunks_collection.query(query_embeddings=[query_embedding], n_results=top_k)
            if results.get("ids"):
                new_chunks = []
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    if chunk_id not in retrieved_ids:
                        retrieved_ids.add(chunk_id)
                        new_chunks.append(results["metadatas"][0][i]["paper_id"])
                if new_chunks:  # Only add non-empty results
                    all_chunks.append("\n".join(new_chunks))
        return all_chunks

# Initialize ChromaDB handler for use in other modules
chroma_db = ChromaDBHandler()
print("✅ ChromaDB retrieval pipeline successfully created!")
