import chromadb as db
import json
from uuid import uuid4
from dotenv import load_dotenv
import os

def ingest(data):
    """
    Ingests data from a specified JSON file into a Chroma database collection named "GoVibes".

    The function loads environment variables to connect to the Chroma database, processes the input JSON data, 
    and adds the documents and metadata into the database with unique identifiers.

    Parameters:
    -----------
    data : str
        The file path of a JSON file containing the data to be ingested into the database.
        The JSON file should be structured such that each item contains a "description" and "data" field.

    Returns:
    --------
    None
        The function does not return any value. It adds the processed data to the Chroma database collection.

    Process:
    --------
    1. Loads environment variables for database connection (e.g., DB_HOST, DB_PORT).
    2. Connects to the Chroma database and creates or retrieves the "GoVibes" collection.
    3. Reads the provided JSON file and extracts the "description" and "data" fields.
    4. Adds the extracted documents and metadata into the Chroma collection with unique UUIDs as identifiers.
    5. Prints the current count of items in the collection after the data has been added.

    Example usage:
    --------------
    ingest("./data/output/augmented/augmented_pois.json")
    """
    load_dotenv()

    client = db.HttpClient(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))

    collection = client.get_or_create_collection(
        "GoVibes", metadata={"hnsw:space": "cosine"}
    )

    with open(data, "r", encoding="utf-8") as file:
        data = json.load(file)

    collection.add(
        documents=list(map(lambda x: x["description"], data)),
        metadatas=list(map(lambda x: x["data"], data)),
        ids=[str(uuid4()) for _ in range(len(data))],
    )

    print("Data lenght:", collection.count())


if __name__ == "__main__":
    ingest("./data/output/augmented/augmented_pois.json")
