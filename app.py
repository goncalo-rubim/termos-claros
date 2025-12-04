from google import genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os

# 1) Carregar variáveis do .env
load_dotenv()

# !!! ALTERAÇÃO 1: Mudar para a variável de ambiente do Google/Gemini !!!
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # !!! ALTERAÇÃO 2: Mudar a mensagem de erro !!!
    raise RuntimeError("GEMINI_API_KEY não está definido no .env")

# 2) Cliente Gemini
# !!! ALTERAÇÃO 3: Usar a biblioteca genai e configurar com a API Key !!!
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    raise RuntimeError(f"Falha ao inicializar o cliente Gemini: {e}")

app = Flask(__name__)

# 3) PROMPT DE SISTEMA
SYSTEM_PROMPT = """
És um assistente especializado em ler e explicar Termos & Condições, Políticas de Privacidade e outros textos legais complexos.

REGRAS GERAIS
- Escreves SEMPRE em português de Portugal.
- NÃO és advogado, NÃO dás aconselhamento jurídico.
- As tuas respostas servem apenas para explicar e simplificar o texto.
- Nunca inventes cláusulas que não estejam no texto fornecido.
- Se algo não estiver claro no texto, dizes explicitamente que não está claro.

OBJETIVO PRINCIPAL
Quando receberes:
- um estilo de resposta pedido
- o texto dos Termos & Condições

Deves:
1) Ler e compreender o texto.
2) Resumir e explicar o que a pessoa está realmente a aceitar.
3) Destacar riscos, obrigações, limitações e pontos sensíveis.
4) Adaptar a linguagem ao ESTILO pedido.
5) Terminar SEMPRE com um aviso de que isto é apenas um resumo simplificado.

ESTRUTURA RECOMENDADA DA RESPOSTA
1) Visão geral rápida
2) O que a empresa recolhe sobre ti
3) Para que usa esses dados
4) Direitos do utilizador
5) Obrigações e comportamentos exigidos
6) Riscos e pontos críticos
7) Aviso final obrigatório

ADAPTAÇÃO AO ESTILO PEDIDO
- ultra simples
- técnico mas claro
- resumo direto e curto
- em tópicos curtos
- estilo desconhecido → tom equilibrado

NUNCA:
- Digas à pessoa se deve aceitar os termos.
- Garantas que os dados estão 100% seguros.
"""
# 4) Função que chama a API do Gemini
def resumir_tc(texto_tc: str, estilo: str) -> str:
    if not estilo:
        estilo = "resumo direto, claro e conciso em linguagem simples"

    user_prompt = f"""
Estilo de resposta pedido: {estilo}

Texto dos Termos & Condições a explicar:
\"\"\"{texto_tc}\"\"\"

Produz UMA resposta única que:
- Siga o ESTILO DE RESPOSTA pedido acima.
- Respeite as REGRAS e a ESTRUTURA definidas na mensagem de sistema.
- Seja fiel ao texto fornecido (não inventes cláusulas).
- Destaque o que o utilizador está a aceitar, os riscos e os direitos.
- Termine com um AVISO claro a dizer que isto é apenas um resumo
  e não é aconselhamento jurídico.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )

    return response.text

# 5) Rota para servir o front-end
@app.route("/")
def home():
    return render_template("index.html")

# 6) Endpoint JSON para o front-end chamar
@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(silent=True) or {}

    texto_tc = data.get("terms_text", "")
    estilo = data.get("style", "").strip()

    if not texto_tc.strip():
        return jsonify({"error": "terms_text em falta"}), 400

    try:
        resumo = resumir_tc(texto_tc, estilo)
        return jsonify({"summary": resumo})

    except Exception as e:
        print("Erro ao chamar a API do Gemini:", repr(e))

        return jsonify({
            "error": "Falha ao gerar o resumo. Verifica a API key, o modelo e a ligação."
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

