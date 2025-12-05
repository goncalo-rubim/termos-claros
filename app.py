import os
import logging
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Configuração de Logging (Crucial para ver erros nos logs do Render)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega .env apenas em local (No Render, as vars vêm do Dashboard)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Permite pedidos de qualquer origem (necessário para frontends separados)

# --- CONFIGURAÇÕES ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# MUDANÇA CRÍTICA 1: Usar 'sonar-pro' para garantir que tabelas e tópicos não quebram.
# O 'sonar' normal tende a ignorar formatação complexa em favor da velocidade.
MODEL_NAME = "sonar-pro"

# --- CONFIGURAÇÃO DE REDE ROBUSTA ---
def get_resilient_session():
    """Cria uma sessão com tentativas automáticas para falhas de rede."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,  # Espera 1s, 2s, 4s entre tentativas
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

http_session = get_resilient_session()

# --- CONSTANTES DE TEXTO ---
AVISO_LEGAL = (
    "> **⚠️ AVISO IA:** Este resumo é gerado automaticamente e serve apenas para fins informativos. "
    "**Não substitui a leitura integral do documento.** Consulte um advogado para decisões legais."
)

STYLE_IDENTITIES = {
    "curto": {
        "role": "Editor Executivo Implacável.",
        "desc": "Estilo telegráfico. Usa apenas listas (bullet points). Frases curtas e secas. Máxima informação, zero gordura. Limite de 200 palavras."
    },
    "detalhado": {
        "role": "Professor de Direito da Universidade.",
        "desc": "Estilo expositivo, claro e minucioso. Explica termos técnicos entre parênteses. Garante que o aluno entende todas as nuances."
    },
    "el5": {
        "role": "Educadora de Infância.",
        "desc": "Usa linguagem muito simples, emojis divertidos e analogias com brinquedos. Proibido usar 'juridiquês'."
    },
    "riscos": {
        "role": "Auditor de Segurança Paranoico.",
        "desc": "Alarmista, crítico e focado apenas no negativo (Red Flags). Encontra todas as armadilhas. Ignora os benefícios."
    },
    "custom": {
        "role": "Assistente Flexível.",
        "desc": "Adapta-se ao pedido do utilizador."
    }
}

# --- ENGENHARIA DE PROMPT OTIMIZADA ---

def construir_mensagens(texto, estilo_key, custom_prompt):
    """
    Constrói as mensagens para a API.
    MUDANÇA CRÍTICA 2: Movemos a estrutura para o USER PROMPT.
    O 'System Prompt' define QUEM a IA é.
    O 'User Prompt' define O QUE e COMO ela deve fazer (no final).
    Isso evita que o componente de busca 'esqueça' a formatação.
    """
    
    # 1. Definir a Persona (System Prompt)
    persona = STYLE_IDENTITIES.get(estilo_key, STYLE_IDENTITIES["curto"])
    
    system_content = (
        f"Tu és: {persona['role']}\n"
        f"O teu modus operandi é: {persona['desc']}\n"
        "O teu idioma de resposta é EXCLUSIVAMENTE Português de Portugal (pt-PT)."
    )

    if estilo_key == "custom" and custom_prompt:
        system_content += f"\nINSTRUÇÃO EXTRA: {custom_prompt}"

    # 2. Definir a Estrutura Rigorosa (User Prompt Suffix)
    # Injetamos isto NO FINAL da mensagem do utilizador para garantir a obediência.
    estrutura_obrigatoria = f"""
    
    ---
    
    ⚠️ **INSTRUÇÕES DE RESPOSTA OBRIGATÓRIAS (NÃO IGNORAR):**
    
    1. A tua resposta TEM de começar com este aviso exato:
       "{AVISO_LEGAL}"
    
    2. Segue ESTRITAMENTE esta estrutura visual (usa Markdown):
    
       **🎯 Resumo Global**
       (Escreve aqui no teu estilo definido)
    
       **🚨 Pontos Críticos**
       (Usa bullet points ou tabelas se necessário)
    
       **👤 Os teus Dados**
       (Como os dados são usados)
    
       **⚖️ Os teus Direitos**
       (O que o utilizador pode fazer)
    
       **💡 Veredito**
       (A tua conclusão final)
       
    3. REGRA VISUAL: Usa tabelas Markdown sempre que houver comparações ou listas de valores.
    """

    if estilo_key == "custom" and custom_prompt:
        # Se for custom, relaxamos a estrutura mas mantemos o aviso
        user_content_final = f"Texto para analisar:\n\n{texto}\n\nLEMBRETE: {custom_prompt}\nComeça com o aviso legal: {AVISO_LEGAL}"
    else:
        user_content_final = f"Texto para analisar:\n\n{texto}\n{estrutura_obrigatoria}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content_final}
    ]

# --- FUNÇÃO DE CHAMADA À API ---

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        logger.error("API Key não encontrada.")
        raise RuntimeError("Configuração do servidor incompleta (API Key).")

    messages = construir_mensagens(texto, estilo_key, custom_prompt)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,       # Temperatura baixa para maior adesão à estrutura
        "max_tokens": 4000,       # Aumentado para permitir tabelas longas
        "top_p": 0.9,
        "frequency_penalty": 0,   # Importante ser 0 para não partir formatação repetitiva (tabelas)
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Timeout de 60s é vital para o 'sonar-pro' ou textos longos
        response = http_session.post(PERPLEXITY_URL, json=payload, headers=headers, timeout=60)
        
        # Log de erro detalhado se não for 200 OK
        if response.status_code != 200:
            logger.error(f"Erro Perplexity {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        logger.error("Timeout na API Perplexity (mais de 60s).")
        raise RuntimeError("A análise demorou demasiado tempo. Tenta um texto mais curto.")
    except Exception as e:
        logger.exception("Erro inesperado na chamada API.")
        raise RuntimeError(f"Erro ao processar: {str(e)}")


# --- ROTAS FLASK ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(silent=True) or {}
    texto = data.get("terms_text", "")
    estilo = data.get("style", "curto")
    custom = data.get("custom_prompt", "")

    # Validações
    if not texto or len(texto.strip()) < 10:
        return jsonify({"error": "Texto demasiado curto."}), 400
    
    if len(texto) > 200000: # Sonar-pro aguenta mais contexto
        return jsonify({"error": "Texto excede o limite de caracteres."}), 400

    try:
        resumo = chamar_perplexity(texto, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        # Retorna erro JSON limpo para o frontend não crashar
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # AVISO: No Render, este bloco é ignorado. O Gunicorn é quem manda.
    app.run(host='0.0.0.0', port=5000, debug=True)
