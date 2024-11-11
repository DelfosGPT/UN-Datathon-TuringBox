prompt_template = """

Basado en el contexto que te voy a proporcionar, trata de obtener mas informacion acerca de un turista, para aumentar la informacion que se tiene sobre su perfil.

Puedes peguntar para aumentar en aspectos como :

 ["Restaurantes", "Parques", 
        "Bares", "Discotecas", "Cines", "Teatros", "Jardines",
        "Museos", "Parques temático", 
        "Parques de atracciones", "Parques acuaticos"]

contexto : "{contexto}"

# Ejemplo 1:
## Ejemplo contexto input : "soy un turista que viaja con su familia y tiene interes en lugares gastronomicos"
## Ejemplo output:
Sobre la cultura ... 
1. Disfruto de los museos y las exposicionesde arte
2. Me interesan las obras teatrales 
3. Me gustan las presentaciones musicales
 
# Ejemplo 2:
## Ejemplo contexto input : "soy un turista que viaja con sus amigos, tengo intereses en cultura, disfruto de los museos"
## Ejemplo output:
Sobre la comida ...
1. Disfruto de la alta cocina
2. Me interesa experimentar con comida tipica
3. Me gustan los restaurantes de comida rapida

# Ejemplo 3:
## Ejemplo contexto input : "soy un turista que viaja con su pareja y tiene interes en lugares gastronomicos, me gusta la alta cocina"
## Ejemplo output:
¿Qué otras actividades te interesan?
1. Disfruto de los parques tematicos
2. Me gustan los teatros y el arte
3. Me interesa conocer bares y discotecas

# Ejemplo 4:
## Ejemplo contexto input : "soy un turista que viaja solo"
## Ejemplo output:
¿Qué actividades te interesan?
1. Me encanta la gastronomía
2. Me disfruto de los parques y jardines
3. Me interesa el arte y la cultura


# Notas Importantes:
- No des las opciones como preguntas, dalas en primera persona en parte del usuario 
- Retorna unicamente una pregunta, como ¿Qué actividades te interesan? y las tres opciones de respuesta.
- Recuerda obtener informacion faltante, preguntar por gastronomia, cultura, e intereses en actividades de ocio sobre las que aún no tengas información en el contexto
"""




import pandas as pd
import json
import time
from botocore.exceptions import ClientError
import boto3
import re



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


def get_user_preference(options):
    while True:
        try:
            print("\nPor favor selecciona una opción (1, 2 o 3):\n")
            selection = int(input())
            if 1 <= selection <= 3:
                return options[selection - 1]
            else:
                print("Por favor ingresa un número entre 1 y 3")
        except ValueError:
            print("Por favor ingresa un número válido")

def extract_options(response):
    lines = response.strip().split('\n')
    options = []
    for line in lines:
        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
            options.append(line.split('. ')[1].strip())
    return options

def main():
    # Contexto inicial
    contexto = "Soy un turista que viaja con ..."
    compania = input(f"{contexto[:-3]}\n 1. Familia\n 2. Amigos\n 3. Pareja\n 4. Solo\nSelecciona una opción (1, 2, 3 o 4): ")

    dict_compania = {
        "1": "familia",
        "2": "amigos",
        "3": "pareja",
        "4": "solo"
    }

    contexto = f"{contexto} {dict_compania[compania]}"
    
    # Realizar tres iteraciones
    for i in range(3):
        print(f"\n=== Iteración {i+1} ===")
        
        # Preparar el prompt con el contexto actual
        prompt = prompt_template.format(contexto=contexto)
        
        # Obtener respuesta de Claude
        response = invoke_claude(prompt)
        print(response)
        
        # Extraer las opciones de la respuesta
        options = extract_options(response)
        
        # Obtener la selección del usuario
        selected_preference = get_user_preference(options)
        
        # Actualizar el contexto con la nueva información
        contexto = f"{contexto}. {selected_preference}"
        
        if i < 2:  # No mostrar este mensaje en la última iteración
            print("\nGenerando la siguiente pregunta...")

    print("\n=== Perfil final del turista ===")
    print(contexto)

if __name__ == "__main__":
    main()