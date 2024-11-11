"""
This script processes a CSV file containing data related to places, invokes the Claude model from Amazon Bedrock 
to generate responses based on a prompt template, and saves the responses to a JSON file.

The main steps involved are:
1. Reading a CSV file containing place data.
2. Constructing a prompt for the model using a template and the data from each row of the CSV.
3. Sending the prompt to the Claude model hosted on Amazon Bedrock for processing.
4. Handling retries and rate-limiting using exponential backoff in case of throttling.
5. Saving the responses in a JSON file to persist progress and avoid reprocessing.

Modules Used:
- pandas: For reading and processing the CSV data.
- json: For reading/writing JSON data.
- boto3: AWS SDK for Python to interact with Amazon Bedrock.
- botocore.exceptions: For handling AWS-specific errors like throttling.
- time: For introducing delays between requests and retries.

Files:
- "./data/input/templates/prompt_template.txt": Template for creating the prompt.
- "./data/input/templates/expected_output_template.txt": Template for the expected output format.
- "./data/output/places/final_places.csv": CSV file with place data to be processed.
- "all_responses.json": JSON file to store the responses from the model.

Functions:
---------
- `invoke_claude(prompt, max_retries=5, initial_delay=1)`: 
    Invokes the Claude model with a given prompt and handles retry logic in case of throttling.
    
Process:
--------
1. Load the CSV file and read the data into a pandas DataFrame.
2. Read the prompt template and expected output template from text files.
3. Set up the AWS Bedrock client with credentials stored in environment variables.
4. Iterate over the rows in the CSV (skipping already processed ones), generate a prompt using the template, 
   and send the prompt to the Claude model for processing.
5. Parse the model's response, extract relevant information, and save it to the `all_responses.json` file.
6. If an error occurs during processing, retry the request up to a specified number of attempts.

Retries and Delay:
-----------------
If the request is rate-limited (throttled), the script will wait and retry the request using exponential backoff. 
It will attempt to process each row a maximum of 5 times before moving to the next row.

Example usage:
--------------
1. Place the required CSV and template files in the correct directory.
2. Ensure AWS credentials are set in the environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
3. Run the script to start processing the rows in the CSV and generating model responses.

Notes:
------
- Make sure to adjust the `region_name` and credentials as needed for your specific AWS setup.
- The script will continue from the last successfully processed row to avoid reprocessing data.
"""


import pandas as pd
import json
import time
from botocore.exceptions import ClientError
import boto3

import os

with open("./data/input/templates/pois_augmentation/prompt_template.txt", 'r', encoding='utf-8') as file:
    prompt_template = file.read()

with open("./data/input/templates/pois_augmentation/expected_output_template.txt", 'r', encoding='utf-8') as file:
    expected_output_template = file.read()


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
