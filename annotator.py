import gradio as gr
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import BlipProcessor, BlipForQuestionAnswering
import numpy as np
import json

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load Sentence Transformers model
similarity_model = SentenceTransformer("paraphrase-distilroberta-base-v1")

# Function to generate JSON response for each question-target pair
def generate_json_response(image, q_t_pairs):
    # Convert input JSON to list of dictionaries
    q_t_pairs_list = json.loads(q_t_pairs)
    
    print("------------------------------")
    print(q_t_pairs_list)
    print("------------------------------")

    # Convert Gradio image data to PIL Image
    image = Image.fromarray(image)
    image = image.convert('RGB')

    # Initialize list to store JSON responses
    responses = []

    # Process each question-target pair
    for pair in q_t_pairs_list:
        # Extract question and target from pair
        question = pair["description"]
        expected_value = pair["value"]

        # Preprocess inputs
        inputs = processor(image, question, return_tensors="pt")

        # Generate answer based on question and image
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        # Compute semantic similarity score
        semantic_score = util.cos_sim(similarity_model.encode([answer]), similarity_model.encode([expected_value]))[0][0]

        # Check if the semantic similarity score is greater than 0.5
        if semantic_score > 0.5:
            response = {expected_value: "true"}
        else:
            response = {expected_value: "false"}

        responses.append(response)
    return json.dumps(responses)

# Create Gradio Interface
input_image = gr.Image(label="Upload an image")
input_q_t_pairs = gr.Textbox(label="Enter question-target pairs JSON", type="text")
output_json = gr.Textbox(label="JSON responses")

iface = gr.Interface(fn=generate_json_response, inputs=[input_image, input_q_t_pairs], outputs=output_json, title="BLIP JSON Response Generation")
iface.launch()

# q_t_pairs = """[
#     {
#         "description": "Is the person in the image standup?",
#         "value": "standup"
#     },
#     {
#         "description": "Can you see the hands of the person?",
#         "value": "hands"
#     },
#     {
#         "description": "Is it inside or outside?",
#         "value": "inside"
#     }
# ]"""
# q_t_pairs_list = json.loads(q_t_pairs)
