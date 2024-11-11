import chromadb as db
import pandas as pd
import random
from dotenv import load_dotenv
import os

def query(profile:str):
    """
    Queries a database of points of interest (POIs) from the GoVibes collection based on the provided user profile,
    and filters the results by random selection of three neighborhoods (comunas) with weighted probabilities.

    The function loads environment variables to connect to the database, retrieves relevant POI data, 
    categorizes the POIs, and saves the top 3 POIs per comuna in a CSV file.

    Parameters:
    -----------
    profile : str
        A description of the user profile that is used to search for relevant POIs in the database.

    Process:
    --------
    1. Loads environment variables for database connection.
    2. Connects to the database and accesses the "GoVibes" collection.
    3. Randomly selects 3 neighborhoods based on specified weighted probabilities.
    4. Queries the database for POIs within each of the selected neighborhoods that match the given profile.
    5. Extracts relevant metadata (e.g., address, categories, rating) from the query results.
    6. Sorts and filters the POIs based on specific criteria (distance and rating).
    7. Saves the final list of top 3 POIs per comuna to a CSV file.
    """
    load_dotenv()

    client = db.HttpClient(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
    collection = client.get_collection("GoVibes")

    probabilities = {
        "Popular": 0.078,
        "Santa Cruz": 0.078,
        "Manrique": 0.078,
        "Aranjuez": 0.078,
        "Castilla": 0.078,
        "Doce de Octubre": 0.078,
        "Robledo": 0.078,
        "Villa Hermosa": 0.078,
        "Buenos Aires": 0.078,
        "La Candelaria": 0.005,
        "Laureles-Estadio": 0.005,
        "La América": 0.078,
        "San Javier": 0.078,
        "El Poblado": 0.004,
        "Guayabal": 0.078,
        "Belén": 0.05,
    }

    keys = list(probabilities.keys())
    weights = list(probabilities.values())

    comunas = []
    while len(comunas) < 3:
        chosen = random.choices(keys, weights=weights, k=1)[0]
        if chosen not in comunas:
            comunas.append(chosen)

    df_poi = pd.DataFrame(
        columns=[
            "ids",
            "documents",
            "distances",
            "address",
            "categories",
            "comuna",
            "latitude",
            "longitude",
            "name",
            "precio",
            "rating bayesian",
        ]
    )
    for comuna in comunas:

        poi = collection.query(
            query_texts = profile,
            n_results = 10,
            where={"comuna": comuna},
        )

        df_temp = pd.DataFrame(
            {
                "ids": poi["ids"][0],
                "documents": poi["documents"][0],
                "distances": poi["distances"][0],
                "address": list(map(lambda x: x["address"], poi["metadatas"][0])),
                "categories": list(map(lambda x: x["categories"], poi["metadatas"][0])),
                "comuna": list(map(lambda x: x["comuna"], poi["metadatas"][0])),
                "latitude": list(map(lambda x: x["latitude"], poi["metadatas"][0])),
                "longitude": list(map(lambda x: x["longitude"], poi["metadatas"][0])),
                "name": list(map(lambda x: x["name"], poi["metadatas"][0])),
                "precio": list(map(lambda x: x["precio"], poi["metadatas"][0])),
                "rating bayesian": list(
                    map(lambda x: x["rating bayesian"], poi["metadatas"][0])
                ),
            }
        ).sort_values("distances", ascending=False)

        df_poi = pd.concat([df_poi, df_temp])
        
    categories = ["restaurante", "parque", "bar", "discoteca", "cine", "teatro", "jardin",
        "museo", "centro comercial"]

    def extract_category(text, categories):
        for category in categories:
            if category in text.lower():
                return category
        return None

    df_poi["final_category"] = df_poi["categories"].apply(lambda x: extract_category(x, categories))
    
    df_final = (
        df_poi.sort_values(["comuna", "distances", "rating bayesian"], ascending=False)
        .drop_duplicates(subset=["comuna", "final_category"])
        .groupby('comuna')
        .head(3)
    )
    path = "./data/output/recomended_pois/top3_places.csv"
    df_final.to_csv(path, index=False)


if __name__ == "__main__":
    query("Soy un turista que viaja con ... amigos. Disfruto visitando museos y galerías de arte. Disfruto probar restaurantes de alta cocina y platos gourmet. Disfruto de ir a bares y discotecas con mis amigos")