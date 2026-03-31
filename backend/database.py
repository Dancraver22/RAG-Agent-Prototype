import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import io

# Define global variables but don't initialize them yet
_ef = None
_client = None

def get_db_resources():
    """Initializes resources only when first requested."""
    global _ef, _client
    if _ef is None:
        # Initializing the embedding function inside the function 
        # prevents it from running during the Render 'Port Scan'
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
    if _client is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
    return _client, _ef

def index_any_csv(file_content: bytes, filename: str):
    """Dynamically indexes CSV rows into ChromaDB."""
    try:
        client, ef = get_db_resources() # Get (or initialize) resources
        df = pd.read_csv(io.BytesIO(file_content))
        
        collection = client.get_or_create_collection(
            name="user_data_vault", 
            embedding_function=ef
        )

        documents = []
        metadatas = []
        ids = []

        for i, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(row_str)
            metadatas.append({"source": filename, "row_index": i})
            ids.append(f"{filename}_{i}")

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return {"status": "success", "rows_indexed": len(documents), "columns": list(df.columns)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_data_vault(query: str, n_results: int = 5):
    """Retrieves relevant rows for the LLM context."""
    try:
        client, ef = get_db_resources() # Get (or initialize) resources
        collection = client.get_collection(name="user_data_vault", embedding_function=ef)
        results = collection.query(query_texts=[query], n_results=n_results)
        return "\n".join(results['documents'][0])
    except:
        return "No relevant data found in the vault."
