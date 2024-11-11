prompt_template = """

Necesito generar un JSON con la siguiente estructura:
1. Un campo 'description' que contenga una descripción comercial de máximo 75 palabras que incluya:
   - Tipo de lugar y ambiente general
   - Servicios disponibles
   - Ocasión ideal de visita y público objetivo
   - Nivel de precios
   - Características especiales del espacio
   La descripción debe ser natural y fluida, sin mencionar nombres, ubicaciones específicas, datos numéricos exactos ni horarios.

2. Un campo 'data' que contenga:
   - name
   - comuna
   - categories (Categorias que describen la tipologia y actividad del punto de interes)
   - address (limpia la dirección para que tenga un formato correcto y sentido, eliminando caracteres extraños o repeticiones)
   - rating bayesian (bayesian_mean)
   - latitude
   - longitude
    - precio (precio entre valores $, $$, $$$, $$$$) 


Notas importantes:
- La descripción debe basarse únicamente en la información proporcionada en el input
- No inventar información adicional
- Mantener el formato JSON consistente
- Los valores numéricos deben mantener su formato original (decimales, etc.)
- La dirección debe limpiarse para mostrar solo una dirección con sentido y formato correcto


{expected_output}

Genera el JSON basado en el siguiente input:

Ten en cuenta que los nombres de las columnas son los siguientes:
"{columns}"

input : "
{input_text}
"
"""

expected_output_template = """
---
## Ejemplo 1:

### Input Ejemplo 1:

,name,desc,score,c_score,price,category,accessibility,schedule,web,search_parameters,phone,address,lat,lon,bayesian_mean,Comuna
0,Los Fabio's Popular,"Información  Opciones de servicio

Asientos al aire libre

Entrega a domicilio

Para llevar

Consumo en el lugar
    Qué ofrece

Comidas durante la madrugada
    Opciones del local

Cena

Espacio con asientos
    Ambiente

Agradable

Informal
    Público usual

Grupos
    Menores

Ideal para ir con niños",4.5,74.0,$ 10.000-20.000,Hamburguesería,,"6 pm.,9 pm.",,Comuna Popular Restaurantes,319 6117987 ,"Cra. 42c #107-001, La Isla, Medellín, Popular, Medellín, Antioquia",6.295462,-75.5485003,4.440476190476191,Popular

### output ejemplo 1:
   
   {
      "description": "hamburguesería informal con ambiente agradable, perfecta para disfrutar en familia o grupos pequeños. Ofrece opciones de servicio flexibles incluyendo consumo en el local, para llevar y delivery. Cuenta con agradable espacio al aire libre y área interior con asientos. Ideal para cenas casuales y antojos nocturnos con precios moderados, siendo una excelente opción para experiencias gastronómicas relajadas.",
      "data": {
         "name": "Los Fabio's Popular",
         "comuna": "Popular",
         "categories": "restaurante, comida rapida, hamburguesas",
         "address": "Cra. 42c #107-001, La Isla, Medellín, Popular, Medellín, Antioquia",
         "rating bayesian": 4.440,
         "latitude": 6.295462,
         "longitude": -75.5485003,
         "precio" : "$"
      }
   }

## Ejemplo 2:

### Input Ejemplo 2:

,name,desc,score,c_score,price,category,accessibility,schedule,web,search_parameters,phone,address,lat,lon,bayesian_mean,Comuna
1390,Museo de Arte Moderno de Medellín,"Información  Museo de arte moderno con una colección permanente y exposiciones rotativas, además de una gran sala de cine.
    Accesibilidad

Entrada accesible para personas en silla de ruedas

Estacionamiento accesible para personas en silla de ruedas

Sanitarios accesibles para personas en silla de ruedas
    Opciones de servicio

Servicios en el lugar
    Servicios

Restaurante

Sanitario
    Menores

Ideal para ir con niños",4.7,11539.0,,Museo,Accesible con silla de ruedas,,http://www.elmamm.org/,Comuna Guayabal Museos,(604) 4442622 ,"Cra. 44 #19a-100, El Poblado, Medellín, El Poblado, Medellín, Antioquia",6.2237797,-75.57384,4.699393886916616,Guayabal

### output ejemplo 2:

el output debe ser:
   {
   "description": "Museo de arte contemporáneo que alberga una fascinante colección permanente y exposiciones temporales, complementado con una sala de cine. Espacio cultural totalmente accesible con instalaciones adaptadas para visitantes con movilidad reducida. Cuenta con servicios de restaurante y comodidades esenciales. Ideal para visitas culturales familiares y educativas.",
   "data": {
      "name": "Museo de Arte Moderno de Medellín",
      "comuna": "El Poblado",
      "categories": "museo, cultura, arte",
      "address": "Cra. 44 #19a-100, El Poblado, Medellín, Antioquia",
      "rating bayesian": 4.699,
      "latitude": 6.2237797,
      "longitude": -75.57384,
      "precio" : "$$"
   }
}


"""




import pandas as pd
import json
import time
from botocore.exceptions import ClientError
import boto3

import os


# Crear el cliente de Bedrock
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',  # Especifica tu región
    aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY')
)


def invoke_claude(prompt, max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 2000,
                "temperature": 1,
                "top_p": 1
            })

            response = bedrock.invoke_model(
                modelId="anthropic.claude-v2",
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['completion']

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    wait_time = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            raise  # Re-raise the exception if it's not a ThrottlingException or we're out of retries

# Read the CSV
df = pd.read_csv('./data/output/places/final_places.csv', index_col=0)

all_responses = []
count = 0

# Load existing responses if the file exists
try:
    with open("all_responses.json", "r") as file:
        all_responses = json.load(file)
        count = len(all_responses)
        print(f"Loaded {count} existing responses")
except FileNotFoundError:
    pass

# Process remaining rows
for index, row in df.iloc[count:].iterrows():
    attempt = 0  # Inicializamos el contador de intentos
    max_attempts = 5  # Número máximo de intentos

    while attempt < max_attempts:
        try:
            columns = ', '.join(df.columns)
            input_text = ', '.join([str(x) for x in row])
            prompt = prompt_template.format(columns=columns, input_text=input_text, expected_output=expected_output_template)
            
            response = invoke_claude(prompt)
            
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}')
            json_response = response[start:end+1]
            json_response = json.loads(json_response)
            
            all_responses.append(json_response)
            
            # Save progress after each successful response
            with open("all_responses.json", "w") as file:
                json.dump(all_responses, file)
            
            count += 1
            print(f"Processed {count}/{len(df)} rows")
            
            # Add a small delay between requests to avoid rate limiting
            time.sleep(1)
            break  # Sale del bucle while si la solicitud fue exitosa

        except Exception as e:
            print(f"Error processing row {count}: {str(e)}")
            attempt += 1  # Incrementamos el número de intentos
            if attempt < max_attempts:
                print(f"Retrying in 5 seconds... Attempt {attempt}/{max_attempts}")
                time.sleep(5)  # Esperamos 5 segundos antes de intentar nuevamente
            else:
                print("Max attempts reached. Moving to the next row.")
                # Save progress before moving to the next row
                with open("all_responses.json", "w") as file:
                    json.dump(all_responses, file)
                break  # Sale del bucle while después de alcanzar el máximo de intentos

print("Processing completed!")
