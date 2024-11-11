import requests
import pandas as pd
import os

def get_transit_directions(origin, destination, api_key):
    """
    Fetches transit directions between an origin and destination using the Google Maps API.

    Parameters:
    - origin (tuple): Latitude and longitude of the starting point as (lat, lon).
    - destination (tuple): Latitude and longitude of the ending point as (lat, lon).
    - api_key (str): API key for Google Maps API.

    Returns:
    - dict: JSON response from the Google Maps API containing route information.
    """
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "transit",
        "language": "es",  # Spanish language for the response
        "key": api_key
    }
    
    response = requests.get(base_url, params=params)
    return response.json()

# Load recommended points of interest (POIs) from CSV
df = pd.read_csv("data/output/recomended_pois/top3_places.csv")

# Select the top 3 places from the dataframe
df = df.head(3)

# Extract routes and names for easier iteration
complete_route = df[["latitude", "longitude"]].values.tolist()
nombres = df["name"].values.tolist()

# Your API key (ensure security by not hardcoding in production)
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Iterate over the route points to fetch and display directions
for i in range(len(complete_route) - 1):
    origin = complete_route[i]
    destination = complete_route[i + 1]
    nombre_origen = nombres[i]
    nombre_destino = nombres[i + 1]

    # Fetch transit directions
    result = get_transit_directions(origin, destination, api_key)

    # Check API response status
    if result["status"] == "OK":
        # Extract route details for display
        route = result["routes"][0]["legs"][0]

        print(
            f"""
        ---     Desde : {nombre_origen}    ---
        ---     Hacia : {nombre_destino}    ---
            """
        )
        print(f"Duración total: {route['duration']['text']}")
        print(f"Distancia total: {route['distance']['text']}")
        
        print("\nPasos del viaje:")
        for step in route["steps"]:
            # Display travel instructions
            print(f"\n- {step['html_instructions']}")
            
            # Display transit details if available
            if "transit_details" in step:
                transit = step["transit_details"]
                print(f"  Línea: {transit['line']['short_name'] if 'short_name' in transit['line'] else transit['line']['name']}")
                print(f"  Desde: {transit['departure_stop']['name']}")
                print(f"  Hasta: {transit['arrival_stop']['name']}")
