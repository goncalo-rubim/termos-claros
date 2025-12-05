import os
import requests
import hashlib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar-pro"

# --- CACHE EM MEMÓRIA (Poupa dinheiro e tempo) ---
# Guarda as últimas 100 respostas. Se o texto for igual, devolve logo.
RESPONSE_CACHE = {}

def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

http_session = get_session()

AVISO_LEGAL = (
    "> **⚠️ AVISO IA:** Este resumo é gerado automaticamente e serve apenas para fins informativos. "
    "**Não substitui a leitura integral do documento nem constitui aconselhamento jurídico profissional.** "
    "Para decisões legais, consulte um advogado."
)

STYLE_IDENTITIES = {
    "curto": "Editor Executivo. Sê direto. Usa APENAS bullet points e tabelas para valores. Máxima síntese.",
    "detalhado": "Professor de Direito. Explica todas as nuances, exceções e termos técnicos.",
    "el5": "Educadora de Infância. Explica como se eu tivesse 5 anos. Usa emojis e metáforas simples.",
    "riscos": "Auditor de Risco. Ignora as coisas boas. Lista APENAS os perigos, multas e abusos de dados."
}

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("API Key não configurada.")

    # 1. VERIFICAR CACHE
    # Criamos um hash único baseada no texto + estilo + prompt extra
    cache_key = hashlib.md5(f"{texto}-{estilo_key}-{custom_prompt}".encode()).hexdigest()
    
    if cache_key in RESPONSE_CACHE:
        print(f"⚡ Cache Hit! Retornando resposta salva para {cache_key[:8]}...")
        return RESPONSE_CACHE[cache_key]

    # 2. SE NÃO ESTIVER EM CACHE, CHAMA A API
    persona = STYLE_IDENTITIES.get(estilo_key, STYLE_IDENTITIES["curto"])
    
    system_content = (
        "Tu és uma IA de análise jurídica ('Termos Claros').\n"
        f"A tua Persona: {persona}\n"
        "Idioma Obrigatório: Português de Portugal (PT-PT)."
    )

    user_content = (
        f"Analisa este texto:\n\n{texto[:100000]}\n\n"
        "--- INSTRUÇÕES FINAIS ---\n"
        "1. Começa com: '> **⚠️ Nota:** Resumo gerado por IA. Não substitui um advogado.'\n"
        "2. Usa Markdown rico (## Títulos, **Negrito**).\n"
        "3. Se houver custos, multas ou prazos, **CRIA OBRIGATORIAMENTE UMA TABELA**.\n"
    )

    if custom_prompt:
        user_content += f"\nATENÇÃO EXTRA AO PEDIDO: {custom_prompt}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 3500
    }

    try:
        response = http_session.post(
            PERPLEXITY_URL, 
            json=payload, 
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            timeout=90
        )
        response.raise_for_status()
        resultado = response.json()["choices"][0]["message"]["content"]
        
        # 3. GUARDAR NO CACHE (Limita a 100 itens para não encher a memória do Render)
        if len(RESPONSE_CACHE) > 100:
            RESPONSE_CACHE.pop(next(iter(RESPONSE_CACHE))) # Remove o mais antigo
        RESPONSE_CACHE[cache_key] = resultado
        
        return resultado

    except Exception as e:
        print(f"Erro API: {e}")
        raise RuntimeError(f"Erro na IA: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(silent=True) or {}
    texto = data.get("terms_text", "")
    estilo = data.get("style", "curto")
    custom = data.get("custom_prompt", "")

    if not texto or len(texto.strip()) < 10:
        return jsonify({"error": "Texto demasiado curto."}), 400

    try:
        resumo = chamar_perplexity(texto, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
