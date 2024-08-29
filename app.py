import os
import json
import re
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from difflib import get_close_matches
import time
import spacy

last_request_time = 0
request_interval = 1

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
conversation_history = []


nlp = spacy.load('en_core_web_sm')

def load_index():
    try:
        with open('C:\\Python\\llm\\docs\\index.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        return None

index_data = load_index()
if index_data is None:
    print("Warning: index.json could not be loaded. The chatbot may not function as expected.")
    index_data = []

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def nlp_process(text):

    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized_text

def find_answer(question, index_data):
    processed_question = nlp_process(question)
    normalized_questions = [(nlp_process(entry['question']), entry['answer']) for entry in index_data]

    closest_match = get_close_matches(processed_question, [q[0] for q in normalized_questions], n=1, cutoff=0.3)

    if closest_match:
        for norm_question, answer in normalized_questions:
            if norm_question == closest_match[0]:
                return answer
    return 'No relevant information found.'

def ask_clarifying_question(question):
  
    if len(question.split()) < 3:  
        return "Could you please provide more details or rephrase your question?"
    return None

def chatbot(question):
    global last_request_time
    current_time = time.time()

    if current_time - last_request_time < request_interval:
        time.sleep(request_interval - (current_time - last_request_time))

    last_request_time = current_time

    clarifying_question = ask_clarifying_question(question)
    if clarifying_question:
        return clarifying_question

    try:
        answer = find_answer(question, index_data)
        if answer:
            return answer

        conversation_history.append({"role": "user", "content": question})

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=conversation_history
        )

        assistant_reply = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Ask a question", placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Answer"),
    title="Chatbot",
    theme="default"
)

interface.launch()
