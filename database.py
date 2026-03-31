import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# 1. Initialize the Client (Stores data in a local folder called 'chroma_data')
client = chromadb.PersistentClient(path="./chroma_data")

# 2. Define the Embedding Function (The 'translator' that turns text into numbers)
# For production, we use a standard lightweight model
default_ef = embedding_functions.DefaultEmbeddingFunction()

def ingest_workout_data(file_path: str):
    """Turns your CSV rows into searchable vectors."""
    df = pd.read_csv(file_path)
    collection = client.get_or_create_collection(name="workout_history", embedding_function=default_ef)

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():
        # Create a text 'story' of the row for the AI to understand
        doc = f"On {row['start_time']}, I did {row['exercise_title']}: {row['weight_kg']}kg for {row['reps']} reps."
        documents.append(doc)
        metadatas.append({"date": str(row['start_time']), "exercise": row['exercise_title']})
        ids.append(f"id_{i}")

    # Add to the Vector DB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return f"Successfully indexed {len(documents)} workout sets."

def query_workouts(query_text: str):
    """Searches memory for relevant workout info."""
    collection = client.get_collection(name="workout_history", embedding_function=default_ef)
    results = collection.query(query_texts=[query_text], n_results=5)
    return results['documents']
