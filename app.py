import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar"

# --- DEFINIÇÃO DE IDENTIDADES (O segredo para funcionar bem) ---
# Em vez de regras complexas, definimos "Quem é a IA" neste momento.
STYLE_IDENTITIES = {
    "curto": (
        "IDENTIDADE: És um Gestor Executivo sem tempo. "
        "ESTILO: Telegráfico, direto ao ponto, usa apenas bullet points. "
        "OBJETIVO: Resumir o máximo de informação no mínimo de palavras."
    ),
    "detalhado": (
        "IDENTIDADE: És um Jurista Professor. "
        "ESTILO: Claro, educativo e completo. Explica o 'porquê' das coisas. "
        "OBJETIVO: Garantir que o utilizador entende todas as nuances."
    ),
    "el5": (
        "IDENTIDADE: És um Professor da Escola Primária. "
        "ESTILO: Usa linguagem infantil, emojis divertidos e analogias (ex: 'os teus brinquedos', 'as regras da casa'). "
        "OBJETIVO: Explicar conceitos complexos a uma criança de 5 anos. NUNCA uses termos técnicos sem explicar."
    ),
    "riscos": (
        "IDENTIDADE: És um Auditor de Segurança Paranoico. "
        "ESTILO: Alarmista, crítico e focado apenas no negativo. "
        "OBJETIVO: Encontrar todas as armadilhas. Ignora as partes boas do texto."
    ),
    "custom": "IDENTIDADE PERSONALIZADA: Segue esta instrução: "
}

# --- PROMPT MESTRA ---
SYSTEM_PROMPT = """
{identity_instruction}

TAREFA:
Analisa os Termos e Condições fornecidos e traduz para Português de Portugal.

REGRA VISUAL (DIAGRAMAS):
Se houver conceitos complexos (ex: fluxo de dados, hierarquia legal), insere uma tag de imagem para ajudar a explicar: .

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:
1. Inicia SEMPRE com este bloco exato:
   > **⚠️ AVISO IA:** Este resumo é informativo e não substitui aconselhamento jurídico profissional.

2. **🎯 Resumo Global** (Escreve no teu ESTILO de identidade)
3. **🚨 Pontos Críticos** (Escreve no teu ESTILO de identidade)
4. **👤 Os teus Dados** (Escreve no teu ESTILO de identidade)
5. **⚖️ Os teus Direitos** (Escreve no teu ESTILO de identidade)
6. **💡 Veredito** (Escreve no teu ESTILO de identidade)
"""

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("API Key não configurada.")

    # 1. Seleciona a Identidade
    identity = STYLE_IDENTITIES.get(estilo_key, STYLE_IDENTITIES["curto"])
    if estilo_key == "custom" and custom_prompt:
        identity += custom_prompt

    # 2. Monta o Prompt de Sistema
    system_content = SYSTEM_PROMPT.format(identity_instruction=identity)

    # 3. Envia o pedido
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Aplica a tua IDENTIDADE e analisa este texto:\n\n{texto}"}
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }

    try:
        response = requests.post(PERPLEXITY_URL, json=payload, headers={
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"Erro API: {e}")
        raise RuntimeError("Erro ao contactar a IA.")

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
    
    if len(texto) > 150000:
        return jsonify({"error": "Texto demasiado longo."}), 400

    try:
        resumo = chamar_perplexity(texto, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
