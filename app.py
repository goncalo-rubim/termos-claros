import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
MODEL_NAME = "sonar"

# --- 1. AVISO LEGAL (Constante para garantir consistência) ---
AVISO_LEGAL = (
    "> **⚠️ AVISO IA:** Este resumo é gerado automaticamente e serve apenas para fins informativos. "
    "**Não substitui a leitura integral do documento nem constitui aconselhamento jurídico profissional.** "
    "Para decisões legais, consulte um advogado."
)

# --- 2. PERSONAS (Mais fortes e distintas) ---
STYLE_IDENTITIES = {
    "curto": (
        "IDENTIDADE: Editor Executivo Implacável.\n"
        "ESTILO: Telegráfico. Usa apenas listas (bullet points). Frases curtas e secas.\n"
        "OBJETIVO: Máxima informação, zero gordura. Limite de 200 palavras."
    ),
    "detalhado": (
        "IDENTIDADE: Professor de Direito da Universidade.\n"
        "ESTILO: Expositivo, claro e minucioso. Explica termos técnicos entre parênteses.\n"
        "OBJETIVO: Garantir que o aluno (utilizador) entende todas as nuances e exceções."
    ),
    "el5": (
        "IDENTIDADE: Educadora de Infância.\n"
        "ESTILO: Usa linguagem muito simples, emojis divertidos e analogias com brinquedos ou regras da casa.\n"
        "OBJETIVO: Traduzir conceitos complexos para uma criança de 5 anos. Proibido usar 'juridiquês'."
    ),
    "riscos": (
        "IDENTIDADE: Auditor de Segurança Paranoico.\n"
        "ESTILO: Alarmista, crítico e focado apenas no negativo (Red Flags).\n"
        "OBJETIVO: Encontrar todas as armadilhas. Ignora os benefícios. Assume sempre o pior cenário."
    ),
    "custom": "Assistente Flexível."
}

# --- 3. PROMPTS DIFERENCIADOS (O Segredo da Lógica) ---

# PROMPT A: Para estilos padrão (Garante a estrutura bonita de 6 pontos)
SYSTEM_PROMPT_ESTRUTURADO = """
{identity_instruction}

TAREFA:
Analisa os Termos e Condições e traduz para Português de Portugal.

REGRA VISUAL:
Se houver conceitos complexos (ex: fluxo de dados), insere uma tag de imagem: .

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:
1. Inicia SEMPRE com este aviso exato:
   {aviso}

2. **🎯 Resumo Global** (No teu estilo de identidade)
3. **🚨 Pontos Críticos** (No teu estilo de identidade)
4. **👤 Os teus Dados** (No teu estilo de identidade)
5. **⚖️ Os teus Direitos** (No teu estilo de identidade)
6. **💡 Veredito** (No teu estilo de identidade)
"""

# PROMPT B: Para estilo personalizado (Ignora a estrutura se o utilizador pedir)
SYSTEM_PROMPT_LIVRE = """
TAREFA: Analisa os Termos e Condições em Português de Portugal.

REGRA DE OURO (Prioridade Máxima):
Segue ESTRITAMENTE a instrução personalizada abaixo.
Se o utilizador pedir um formato específico (ex: "apenas 5 linhas", "só uma lista"), IGNORA qualquer estrutura padrão e cumpre o pedido do utilizador.

INSTRUÇÃO PERSONALIZADA: {custom_instruction}

REGRA DE SEGURANÇA:
Independentemente do pedido, começa a resposta com este aviso:
{aviso}

REGRA VISUAL:
Usa  se ajudar a explicar.
"""

def chamar_perplexity(texto: str, estilo_key: str, custom_prompt: str = "") -> str:
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("A API Key do Perplexity não está configurada.")

    # LÓGICA DE SELEÇÃO DE PROMPT
    if estilo_key == "custom" and custom_prompt:
        # Se for personalizado, usa o Prompt Livre (sem estrutura fixa)
        system_content = SYSTEM_PROMPT_LIVRE.format(
            custom_instruction=custom_prompt,
            aviso=AVISO_LEGAL
        )
    else:
        # Se for padrão, usa o Prompt Estruturado (com os 6 tópicos)
        identidade = STYLE_IDENTITIES.get(estilo_key, STYLE_IDENTITIES["curto"])
        system_content = SYSTEM_PROMPT_ESTRUTURADO.format(
            identity_instruction=identidade,
            aviso=AVISO_LEGAL
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Texto para analisar:\n\n{texto}"}
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
    except Exception as e:
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
