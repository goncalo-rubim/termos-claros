import os
import requests
import hashlib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)
CORS(app)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar-pro"

# Cache em Memória
RESPONSE_CACHE = {}

# --- REDE ROBUSTA ---
def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

http_session = get_session()

# --- UTILITÁRIOS ---
def extrair_texto_pdf(file_storage):
    try:
        reader = PdfReader(file_storage)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Erro PDF: {e}")
        return None

# --- PROMPTS ---
STYLE_PROMPTS = {
    "curto": "Editor Executivo. Resposta direta, bullet points, tabelas para valores.",
    "detalhado": "Advogado Sénior. Explica nuances, exceções e termos técnicos.",
    "el5": "Educadora de Infância. Explica como se eu tivesse 5 anos (analogias simples).",
    "riscos": "Auditor de Risco. Ignora o positivo. Foca APENAS nos perigos, multas e dados."
}

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("API Key não configurada.")

    # 1. Verificar Cache
    cache_key = hashlib.md5(f"{texto[:5000]}-{estilo_key}-{custom_prompt}".encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]

    # 2. Construir Prompt
    persona = STYLE_PROMPTS.get(estilo_key, STYLE_PROMPTS["curto"])
    
    system_content = (
        "Tu és a IA 'Termos Claros'. O teu objetivo é proteger o consumidor.\n"
        f"Modo: {persona}\n"
        "Idioma: Português de Portugal (PT-PT)."
    )

    # MELHORIA: Instrução explícita para usar Blockquote (>) no aviso
    user_content = (
        f"Analisa este documento ({len(texto)} caracteres):\n\n{texto[:100000]}\n\n"
        "--- REGRAS VISUAIS OBRIGATÓRIAS ---\n"
        "1. A PRIMEIRA COISA a escrever é este bloco de citação exato:\n"
        "   > **⚠️ AVISO LEGAL:** Esta análise é gerada por Inteligência Artificial (Sonar-Pro) para fins informativos. **Não substitui a leitura integral nem o aconselhamento jurídico profissional.**\n\n"
        "2. Depois do aviso, usa Markdown rico (## Títulos, **Negrito**).\n"
        "3. Se houver valores (multas, preços) ou prazos, CRIA UMA TABELA.\n"
    )

    if custom_prompt:
        user_content += f"\nFOCO DO UTILIZADOR: {custom_prompt}"

    # 3. Chamar API
    try:
        response = http_session.post(
            PERPLEXITY_URL, 
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}],
                "temperature": 0.1,
                "max_tokens": 3000
            }, 
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            timeout=90
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        
        # Guardar Cache
        if len(RESPONSE_CACHE) > 50:
            RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
        RESPONSE_CACHE[cache_key] = result
        
        return result

    except Exception as e:
        print(f"Erro API: {e}")
        raise RuntimeError(f"Erro na IA: {str(e)}")

# --- ROTAS ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    texto_final = ""

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            texto_final = extrair_texto_pdf(file)
    elif request.form.get("terms_text"):
        texto_final = request.form.get("terms_text")
    elif request.is_json:
        texto_final = request.get_json().get("terms_text", "")

    if not texto_final or len(texto_final.strip()) < 10:
        return jsonify({"error": "Texto insuficiente ou PDF ilegível."}), 400

    estilo = request.form.get("style") or (request.json.get("style") if request.is_json else "curto")
    custom = request.form.get("custom_prompt") or (request.json.get("custom_prompt") if request.is_json else "")

    try:
        resumo = chamar_perplexity(texto_final, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
