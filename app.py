# app.py - Amani Pro Gratuit
from flask import Flask, request, jsonify, abort
from transformers import pipeline, AutoTokenizer
import torch
import sqlite3
import gzip
import json
import numpy as np
from difflib import get_close_matches
import gc  # Pour gérer la mémoire

app = Flask(__name__)

# Configuration minimale
MODEL_NAME = "distilbart-cnn-12-6"  # Modèle plus léger
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = pipeline("text-generation", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# Token d'authentification (à remplacer par une variable d'environnement ou une config plus sûre)
API_TOKEN = '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'

# Base de connaissances africaine compressée
with gzip.open('african_knowledge.json.gz', 'rt', encoding='utf-8') as f:
    KNOWLEDGE = json.load(f)

# Cache mémoire intelligent
cache = {}

def detect_language(text):
    """Détection ultra-optimisée des langues"""
    if any(c in text.lower() for c in ['é','è','ê']): return 'fr'
    if any(word in text.lower() for word in ['the','and','of']): return 'en'
    if any(word in text.lower() for word in ['jërëjëf','jamm']): return 'wo'
    return 'en'

def generate_response(prompt):
    # Limiter la taille du cache à 100 éléments
    if len(cache) > 100:
        cache.clear()  # Vider le cache si trop d'éléments sont stockés

    # Vérification du cache  
    if prompt in cache:  
        return cache[prompt]  

    # Recherche contextuelle  
    context = next((v for k, v in KNOWLEDGE.items() if k.lower() in prompt.lower()), "")  

    # Génération avec le modèle  
    input_text = f"[Contexte: {context}] {prompt} (Réponds en {detect_language(prompt)} avec style africain)"  

    output = model(  
        input_text,  
        max_length=100,  # Limiter la longueur de la réponse pour économiser la mémoire  
        num_return_sequences=1,  
        temperature=0.7,  
        do_sample=True  
    )[0]['generated_text']  

    # Post-traitement africain  
    if detect_language(prompt) == 'fr':  
        output = f"*Son de balafon* {output.split(']')[-1].strip()} *souffle dans les mains*"  

    # Libération de la mémoire  
    gc.collect()  

    # Stockage dans le cache  
    cache[prompt] = output  
    return output

# Fonction d'authentification
def authenticate():
    token = request.headers.get('Authorization')
    if token != f"Bearer {API_TOKEN}":
        abort(401, description="Unauthorized access")

# API Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    authenticate()  # Vérification du token d'authentification

    prompt = request.json.get('prompt', '')
    response = generate_response(prompt)
    return jsonify({
        "response": response,
        "language": detect_language(prompt)
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
    body { font-family: Ubuntu, sans-serif; background: #f5f5dc; }
    .response { background: #2c3e50; color: white; padding: 15px; border-radius: 10px; }
    </style>
    </head>
    <body>
    <h1>🌍 Amani Pro - IA Africaine</h1>
    <textarea id="input" placeholder="Pose ta question..."></textarea>
    <button onclick="chat()">Envoyer</button>
    <div id="output" class="response"></div>
    <script>
    async function chat() {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': 'Bearer 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'  // Ajouter le token dans l'entête
            },
            body: JSON.stringify({ prompt: document.getElementById('input').value })
        });
        const data = await response.json();
        document.getElementById('output').innerHTML =
        <strong>${data.language.toUpperCase()}:</strong> ${data.response};
    }
    </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
