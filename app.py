import os
import time
import logging
import hashlib
import re
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# --- 1. CONFIGURAÇÃO DE ENGENHARIA ---
load_dotenv()

# Logging Estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s'
)
logger = logging.getLogger("TermosClarosEngine")

app = Flask(__name__)
CORS(app)

# Rate Limiting (Proteção contra abuso: 10 pedidos por minuto por IP)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute", "200 per day"],
    storage_uri="memory://"
)

# Constantes
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
MAX_TOKENS = 4000
MAX_CHAR_LIMIT = 150000

# --- 2. SERVIÇO DE CACHE TTL (Time-To-Live) ---
class SmartCache:
    """Cache LRU com expiração de tempo para gestão eficiente de memória."""
    def __init__(self, capacity: int = 50, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            self.cache.pop(key)
            return None
        self.cache.move_to_end(key)
        return value

    def set(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, time.time())
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

cache_service = SmartCache()

# --- 3. SERVIÇOS DE UTILIDADE ---
class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove espaços extra e caracteres inúteis para poupar tokens."""
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:MAX_CHAR_LIMIT]

    @staticmethod
    def extract_from_pdf(file_stream) -> str:
        try:
            reader = PdfReader(file_stream)
            text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(text)
        except Exception as e:
            logger.error(f"PDF Error: {e}")
            raise ValueError("O ficheiro PDF está corrompido ou é ilegível.")

class AIService:
    def __init__(self):
        self.session = Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate_prompt(self, text: str, style: str, custom: str) -> list:
        personas = {
            "curto": "Sê um Editor Executivo. Resposta telegráfica. Usa APENAS bullet points.",
            "detalhado": "Sê um Jurista Sénior. Explica com profundidade, citando implicações legais.",
            "el5": "Sê um Professor do 1º Ciclo. Explica como se eu tivesse 5 anos (metáforas simples).",
            "riscos": "Sê um Auditor de Risco. Foca 100% nas cláusulas perigosas e abusivas (RED FLAGS)."
        }
        
        persona = personas.get(style, personas["curto"])
        
        system_msg = (
            "Tu és o 'Termos Claros AI', uma inteligência de elite em análise contratual.\n"
            "Idioma de Saída: Português de Portugal (PT-PT).\n"
            f"Persona Ativa: {persona}"
        )

        user_msg = (
            f"Analisa este texto ({len(text)} chars):\n\n{text[:50000]}...\n\n"
            "--- OUTPUT REQUIREMENTS ---\n"
            "1. Inicia com: '> **⚠️ Nota:** Análise gerada por IA (Sonar Pro). Não substitui advogado.'\n"
            "2. Usa formatação Markdown rica (H2, **Bold**, Tabelas).\n"
            "3. Se houver valores monetários ou prazos, OBRIGATÓRIO criar uma tabela.\n"
            "4. Estrutura: Resumo Executivo -> Pontos Críticos -> Análise de Dados -> Veredito.\n"
        )
        
        if custom:
            user_msg += f"\nPRIORIDADE MÁXIMA AO PEDIDO: {custom}"

        return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    def analyze(self, text: str, style: str, custom: str) -> str:
        # Cache Key Generation
        content_hash = hashlib.md5((text[:1000] + style + custom).encode()).hexdigest()
        
        cached = cache_service.get(content_hash)
        if cached:
            logger.info(f"Cache HIT: {content_hash}")
            return cached

        # API Call
        try:
            payload = {
                "model": MODEL,
                "messages": self.generate_prompt(text, style, custom),
                "temperature": 0.1,
                "max_tokens": MAX_TOKENS
            }
            
            resp = self.session.post(
                API_URL, 
                json=payload, 
                headers={"Authorization": f"Bearer {PERPLEXITY_KEY}"},
                timeout=90
            )
            resp.raise_for_status()
            
            result = resp.json()["choices"][0]["message"]["content"]
            cache_service.set(content_hash, result)
            return result
            
        except Exception as e:
            logger.error(f"API Error: {e}")
            raise RuntimeError(f"Falha na comunicação com a IA: {str(e)}")

ai_engine = AIService()

# --- 4. ROTAS DA APLICAÇÃO ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
@limiter.limit("5 per minute") # Proteção extra por IP
def analyze_endpoint():
    try:
        text = ""
        
        # 1. Input Handling
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                filename = secure_filename(file.filename)
                if filename.endswith('.pdf'):
                    text = TextProcessor.extract_from_pdf(file)
                else:
                    return jsonify({"error": "Formato não suportado. Use PDF."}), 400
        
        elif request.form.get('text'):
            text = request.form.get('text')
        
        # 2. Validation
        clean_text = TextProcessor.clean_text(text)
        if len(clean_text) < 50:
            return jsonify({"error": "Texto insuficiente para análise."}), 400

        # 3. Processing
        style = request.form.get('style', 'curto')
        custom = request.form.get('custom', '')
        
        summary = ai_engine.analyze(clean_text, style, custom)
        
        return jsonify({
            "success": True, 
            "summary": summary, 
            "stats": {"chars": len(clean_text), "model": MODEL}
        })

    except Exception as e:
        logger.error(f"Critical Endpoint Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Muitos pedidos. Tenta novamente em 1 minuto."}), 429

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
