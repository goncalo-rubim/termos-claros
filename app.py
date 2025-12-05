import os
import requests
import hashlib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- SEGURANÇA & LIMITES ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["15 per minute", "500 per day"],
    storage_uri="memory://"
)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar-pro"

# --- CACHE INTELIGENTE (LRU Simplificado) ---
# Guarda as últimas 50 respostas. Se enviares o mesmo texto, a resposta é instantânea (0s).
RESPONSE_CACHE = {}

def get_session():
    session = requests.Session()
    # Retry agressivo para garantir que nunca falha no Render
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

http_session = get_session()

STYLE_PROMPTS = {
    "curto": "Editor Chefe. Sê conciso. Usa APENAS bullet points e tabelas.",
    "detalhado": "Advogado Sénior. Explica cada cláusula, exceção e implicação legal.",
    "el5": "Professor do Primário. Explica como se eu tivesse 5 anos (usa analogias simples).",
    "riscos": "Auditor de Segurança. Ignora o positivo. Lista APENAS os perigos, multas e dados partilhados."
}

def extrair_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        # Extrai e limpa o texto (remove quebras excessivas)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Erro PDF: {e}")
        return ""

def chamar_perplexity(texto: str, estilo: str, custom: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("API Key em falta.")

    # 1. Verificar Cache (Hash MD5 do input)
    cache_key = hashlib.md5(f"{texto[:5000]}-{estilo}-{custom}".encode()).hexdigest()
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]

    # 2. Construir Prompt
    persona = STYLE_PROMPTS.get(estilo, STYLE_PROMPTS["curto"])
    
    system_msg = (
        "Tu és a IA 'Termos Claros', especialista em simplificação jurídica.\n"
        f"Modo: {persona}\n"
        "Idioma: Português de Portugal (PT-PT)."
    )

    user_msg = (
        f"Analisa este documento ({len(texto)} caracteres):\n\n{texto[:100000]}\n\n"
        "--- REGRAS VISUAIS ---\n"
        "1. Inicia com: '> **⚠️ Nota:** Análise automática (IA). Consulte um advogado.'\n"
        "2. Usa H2 (Markdown ##) para secções.\n"
        "3. **OBRIGATÓRIO:** Se houver valores (multas, preços) ou prazos, CRIA UMA TABELA.\n"
    )

    if custom:
        user_msg += f"\nFOCO DO UTILIZADOR: {custom}"

    # 3. Chamada API
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            "temperature": 0.1,
            "max_tokens": 3500
        }
        
        resp = http_session.post(
            PERPLEXITY_URL, json=payload, 
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}, 
            timeout=90
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]

        # Guardar em Cache
        if len(RESPONSE_CACHE) > 50:
            RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE)))
        RESPONSE_CACHE[cache_key] = result
        
        return result

    except Exception as e:
        raise RuntimeError(f"Erro na IA: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
@limiter.limit("10 per minute")
def summarize():
    text = ""
    
    # Detetar PDF ou Texto
    if 'file' in request.files:
        text = extrair_pdf(request.files['file'])
    else:
        data = request.get_json(silent=True) or request.form
        text = data.get("terms_text", "")

    if len(text) < 10:
        return jsonify({"error": "Texto insuficiente."}), 400

    style = request.form.get("style", "curto")
    custom = request.form.get("custom_prompt", "")

    try:
        summary = chamar_perplexity(text, style, custom)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
