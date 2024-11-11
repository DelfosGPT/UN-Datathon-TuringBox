"""
This script simulates a conversational process where a user (tourist) answers a series of questions, 
and based on the user's responses, a profile is created using the Claude model hosted on Amazon Bedrock.

The main tasks performed by this script are:
1. Prompting the user for their travel companions (family, friends, partner, solo).
2. Interacting with the Claude model (via Amazon Bedrock) to generate a conversation for further preferences.
3. Parsing and processing model responses, extracting available options for the user to select.
4. Updating the context based on the user's selections and generating a new prompt for the next iteration.
5. Iterating three times to gather comprehensive preferences from the user.
6. Displaying the final tourist profile based on the user’s responses.

Modules Used:
-------------
- `pandas`: Not used in the code but could be useful for future expansion.
- `json`: For handling JSON data and responses from the Claude model.
- `time`: For adding delays during retries in case of throttling from the AWS service.
- `botocore.exceptions`: For handling exceptions specific to AWS (e.g., `ThrottlingException`).
- `boto3`: The AWS SDK for Python, used for invoking the Claude model hosted on Amazon Bedrock.
- `re`: Not explicitly used in the code but may be used for regex processing.
- `os`: For accessing environment variables for AWS credentials.

Files:
------
- `./data/input/templates/user_context/prompt_template.txt`: A text file containing the template for generating prompts based on the user's context.

Functions:
----------
- `invoke_claude(prompt, max_retries=5, initial_delay=1)`:
    Sends a request to the Claude model with the provided prompt, handles throttling using exponential backoff, 
    and returns the model's response (completion).
    
- `get_user_preference(options)`:
    Prompts the user to select an option from the available choices (1, 2, or 3) and returns the selected option.

- `extract_options(response)`:
    Parses the response from Claude, extracting the available options for the user to choose from.

- `main()`:
    Orchestrates the conversation, prompting the user for their initial context and preferences,
    generating prompts for the Claude model, processing responses, and updating the context for further questions.
    
Process:
--------
1. **Initial User Input**: The script starts by asking the user to choose their travel companion (family, friends, partner, or solo).
2. **Prompt Generation and Model Interaction**: For three iterations, the script generates a new prompt based on the user's current profile and sends it to Claude for processing.
3. **User Selection**: Each model response contains options for the user to choose from, and the user selects their preferred option.
4. **Updating the Context**: After each iteration, the context is updated based on the user’s response, which is then passed to the Claude model for the next round of questions.
5. **Final Output**: After three iterations, the user’s complete profile is displayed, reflecting their preferences.

Retries and Delay:
-----------------
- If a throttling error occurs during the interaction with the Claude model, the script will retry the request with exponential backoff (increasing wait time after each attempt) to avoid hitting rate limits.
- The script will make up to 5 retry attempts before giving up on a request.

Example Usage:
--------------
1. Ensure that AWS credentials are set in the environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`).
2. Place the `prompt_template.txt` in the specified directory.
3. Run the script, and it will prompt the user for their travel companion and process their selections.
4. The profile of the tourist will be generated and printed after 3 iterations.

Notes:
------
- Modify the `contexto` variable for more detailed or specific context about the user’s travel profile.
- Ensure that AWS Bedrock is properly configured for use with the `boto3` client.

"""

import pandas as pd
import json
import time
from botocore.exceptions import ClientError
import boto3
import re
import os

with open("./data/input/templates/user_context/prompt_template.txt", 'r', encoding='utf-8') as file:
    prompt_template = file.read()

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
    with open("./data/output/user_profile/profile.txt", "w") as f:
        f.write(contexto)
        print("El perfil del turista se ha generado y guardado en el archivo 'profile.txt'")

if __name__ == "__main__":
    main()