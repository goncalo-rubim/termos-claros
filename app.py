import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configurações
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar"

# --- PERSONALIDADES DA IA ---
STYLE_PROMPTS = {
    "curto": "Sê conciso. Usa apenas bullet points curtos. Foca no essencial. Máximo 200 palavras.",
    "detalhado": "Explica detalhadamente. Se houver conceitos técnicos (ex: cookies, arbitragem), explica-os de forma simples.",
    "el5": "Explica como se eu tivesse 5 anos. Usa analogias do dia-a-dia. Tom educativo.",
    "riscos": "Sê o 'Advogado do Diabo'. Ignora os benefícios. Lista APENAS perigos, renúncias de direitos e dados recolhidos.",
    "custom": "Segue estritamente a instrução personalizada do utilizador."
}

# Prompt de Sistema (Cérebro)
SYSTEM_PROMPT_BASE = """
És o 'Termos Claros', um assistente jurídico AI especializado em Proteção do Consumidor (Portugal/EU).

⚠️ REGRA CRÍTICA DE FORMATO:
A tua resposta DEVE começar SEMPRE com este bloco exato (Markdown quote):

> **⚠️ AVISO IA:** Este resumo é gerado automaticamente e serve apenas para fins informativos. **Não substitui a leitura integral do documento nem constitui aconselhamento jurídico profissional.** Para decisões legais, consulte um advogado.

---

ESTRUTURA DO RESUMO (Usa Markdown):
1. 🎯 **Resumo em 1 Frase**
2. 🚨 **Red Flags & Riscos** (Usa emojis de alerta)
3. 👤 **Os teus Dados** (O que recolhem e com quem partilham)
4. ⚖️ **Os teus Direitos** (Cancelamento, Reembolso, Litígios)
5. 💡 **Veredito Final**

INSTRUÇÃO VISUAL:
Se explicares um fluxo de dados complexo, usa a tag: `

[Image of data flow diagram explaining X]
`.

CONTEXTO: O utilizador pediu o estilo: "{style_instruction}"
"""

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("A API Key do Perplexity não está configurada.")

    # Define a instrução de estilo
    instruction = STYLE_PROMPTS.get(estilo_key, STYLE_PROMPTS["curto"])
    if estilo_key == "custom" and custom_prompt:
        instruction = f"Instrução personalizada: {custom_prompt}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_BASE.format(style_instruction=instruction)},
            {"role": "user", "content": f"Analisa os seguintes Termos & Condições:\n\n{texto}"}
        ],
        "temperature": 0.1, # Baixa temperatura para precisão factual
        "max_tokens": 2500
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
        # Retorna uma mensagem de erro genérica para o frontend não quebrar
        raise RuntimeError("Não foi possível contactar a inteligência artificial. Tente novamente.")

# --- ROTAS ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(silent=True) or {}
    texto = data.get("terms_text", "")
    estilo = data.get("style", "curto")
    custom = data.get("custom_prompt", "")

    # Validações de Backend
    if not texto or len(texto.strip()) < 10:
        return jsonify({"error": "O texto é demasiado curto para ser analisado."}), 400
    
    if len(texto) > 150000:
        return jsonify({"error": "Texto demasiado longo (limite: 150k caracteres)."}), 400

    try:
        resumo = chamar_perplexity(texto, estilo, custom)
        return jsonify({"summary": resumo})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
