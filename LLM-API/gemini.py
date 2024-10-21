import json
import time
import random
import asyncio
from tqdm import tqdm 
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

API_keys_dict = {
    0: "AIzaSyD-iSybhlu9W_OvFU_uKNObNbqjgfKZh9w",
    1: "AIzaSyCsgKDLD3lbd_mIFtMzz2W273aSnAESCQ4",
    2: "AIzaSyAAzHOObpAQ2xVafCi5TSIiijm-fmFrEoM"
}

# Configure the generative AI with the API key
genai.configure(api_key=API_keys_dict[2])

# Load your JSON data
with open('medqa_data.json') as f:
    data = json.load(f)

# Define the generation configuration for the model
generation_config = {
    "temperature": 0.3,
    "top_p": 1.0,
    "top_k": 64,
    "max_output_tokens": 200,
    "response_mime_type": "text/plain",
}

# Define relaxed safety settings
safety_settings_dict = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Create the model with the generation configuration and safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings_dict
)

# Start a chat session with the model
chat_session = model.start_chat(
    history=[]
)

# Create a list to store the responses and track success
responses = []
succeeded = 0

# Loop through each item in the dataset
for item in tqdm(data):
    question = item['question']
    answer = item['answer']
    
    # Construct the prompt for the model
    prompt = f"""
    Given the following medical question and its correct answer, generate a detailed explanatory paragraph suitable for a medical textbook.
    Ensure the explanation is accurate, uses appropriate medical terminology, and is written in a clear and professional tone.
    Question: {question}
    Answer: {answer}
    """
    
    # Attempt to generate the response, handle exceptions gracefully
    try:
        response = chat_session.send_message(prompt)
        succeeded += 1
    except Exception as e:
        print(f"Error generating response for question: {question}.\n Error: {e}.\nSkipping...")
        continue
    
    # Extract and store the response
    response_text = response.text  # Extract the text of the response
    item['detailed_explanation'] = response_text  # Add a new key-value pair to the item
    responses.append(item)
    
    # Wait before sending the next request
    wait_time = 5 + random.uniform(-0.5, 0.5)
    time.sleep(wait_time)

# Convert the updated data to JSON format
updated_data_json = json.dumps(data, indent=2)

print(f"Generated {succeeded}/{len(data)} responses")

# Save the updated JSON data to a file
with open('medqa_data_with_responses_h9w.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)

# Print an example of the updated data
print(f"Example:\n{json.dumps(data[0], indent=2)}\n")
