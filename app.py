# app.py - Amani Pro Gratuit

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sqlite3
import gzip
import numpy as np
import json
from difflib import get_close_matches

app = Flask(__name__)

# Configuration minimale
MODEL_NAME = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_obj = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = pipeline("text2text-generation", model=model_obj, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Base de connaissances africaine compressée
with gzip.open('african_knowledge.json.gz', 'rt', encoding='utf-8') as f:
    KNOWLEDGE = json.load(f)

# Cache mémoire intelligent
cache = {}

def detect_language(text):
    """Détection ultra-optimisée des langues"""
    lower = text.lower()
    if any(c in lower for c in ['é','è','ê']): return 'fr'
    if any(w in lower for w in ['the','and','of']): return 'en'
    if any(w in lower for w in ['jërëjëf','jamm']): return 'wo'
    return 'en'

def generate_response(prompt):
    lang = detect_language(prompt)
    if prompt in cache:
        return cache[prompt]

    # Recherche contextuelle simple
    context = next((v for k, v in KNOWLEDGE.items() if k.lower() in prompt.lower()), "")

    input_text = f"[Contexte: {context}] {prompt} (Réponds en {lang} avec style africain)"
    output = model(
        input_text,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )[0]['generated_text']

    # Post-traitement
    if lang == 'fr':
        output = f"*Son de balafon* {output.split(']')[-1].strip()} *souffle dans les mains*"

    cache[prompt] = output
    return output

# API Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Invalid input, JSON expected"}), 400

    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    lang = detect_language(prompt)
    response = generate_response(prompt)
    return jsonify({
        "response": response,
        "language": lang
    })

# Interface
@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Amani Pro</title>
        <style>
            body { font-family: Ubuntu, sans-serif; background: #f5f5dc; padding: 20px; }
            .response { background: #2c3e50; color: white; padding: 15px; border-radius: 10px; margin-top: 10px; }
            textarea { width: 100%; height: 80px; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🌍 Amani Pro - IA Africaine</h1>
        <textarea id="input" placeholder="Pose ta question..."></textarea><br>
        <button onclick="chat()">Envoyer</button>
        <div id="output" class="response"></div>
        <script>
            async function chat() {
                const prompt = document.getElementById('input').value;
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await response.json();
                document.getElementById('output').innerHTML =
                    `<strong>${data.language.toUpperCase()}:</strong> ${data.response}`;
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
