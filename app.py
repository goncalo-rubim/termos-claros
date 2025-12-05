import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Carrega variáveis do ficheiro .env (apenas para desenvolvimento local)
# No Render, as variáveis são lidas diretamente do sistema.
load_dotenv()

app = Flask(__name__)

# --- CONFIGURAÇÃO ---

# A chave da API deve estar definida nas "Environment Variables" do Render
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Endpoint oficial da API Perplexity
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Modelo escolhido (o mais capaz e com maior contexto atual)
# Nota: A Perplexity atualiza nomes frequentemente, este é o topo de gama atual baseado no Llama 3.1
MODEL_NAME = "sonar"

# Prompt de Sistema (Cérebro da IA)
SYSTEM_PROMPT = """
És um especialista jurídico sênior (mas não advogado) que traduz "legalês" para Português de Portugal claro, estruturado e acessível.

OBJETIVO:
Ler o texto jurídico fornecido e gerar um resumo prático formatado em MARKDOWN.

REGRAS DE FORMATAÇÃO:
- Usa `###` para títulos de secções.
- Usa listas com hífens `-` para facilitar a leitura.
- Usa **negrito** para destacar riscos ou dados sensíveis.
- Não uses blocos de código para o texto normal.

ESTRUTURA DA RESPOSTA:
1. ### 🎯 Resumo em 1 Frase
   (A essência do documento numa frase simples)

2. ### 🚩 Red Flags (Pontos Críticos)
   (Lista com emojis 🔴 para cláusulas perigosas, abusivas, renúncias de direitos ou coisas estranhas)

3. ### 👤 Os teus Dados
   (O que recolhem, cookies, localização, e com quem partilham)

4. ### ⚖️ Os teus Direitos
   (Como cancelar, apagar conta, ou resolver disputas)

5. ### 💡 Conclusão
   (Veredito final neutro)

Termina sempre com:
*Aviso: Isto é um resumo automático gerado por IA e não substitui aconselhamento jurídico profissional.*
"""

def chamar_perplexity(texto: str, estilo: str) -> str:
    """
    Envia o texto para a API da Perplexity e devolve o resumo.
    """
    if not PERPLEXITY_API_KEY:
        raise RuntimeError("A variável de ambiente PERPLEXITY_API_KEY não está configurada.")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construção da mensagem para o Chat Completion
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Estilo de resposta desejado: {estilo}\n\nTexto dos Termos para analisar:\n{texto}"}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,       # Baixa temperatura para reduzir alucinações
        "max_tokens": 3000,       # Limite de resposta (suficiente para resumos detalhados)
        "top_p": 0.9,
        "return_citations": False # Não precisamos de citações da web para analisar um texto colado
    }

    try:
        response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)
        response.raise_for_status() # Lança exceção se o código HTTP for 4xx ou 5xx
        
        data = response.json()
        
        # Extrai o conteúdo da resposta da IA
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição à API: {e}")
        # Tenta obter detalhes do erro se a API devolveu JSON de erro
        if e.response is not None:
             print(f"Detalhe da API: {e.response.text}")
        raise RuntimeError("Falha ao comunicar com a inteligência artificial.")

# --- ROTAS DA APLICAÇÃO ---

@app.route("/")
def home():
    # Serve o ficheiro index.html da pasta 'templates'
    return render_template("index.html")

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    # Obtém os dados JSON enviados pelo frontend
    data = request.get_json(silent=True) or {}
    
    texto_tc = data.get("terms_text", "")
    estilo = data.get("style", "claro e direto")

    # 1. Validação: Texto vazio ou muito curto
    if not texto_tc or len(texto_tc.strip()) < 10:
        return jsonify({"error": "O texto fornecido é demasiado curto. Por favor, cola o texto completo."}), 400
    
    # 2. Validação: Texto excessivamente longo (Segurança)
    # 120.000 caracteres é um limite seguro para evitar sobrecarregar o servidor/API
    if len(texto_tc) > 120000:
        return jsonify({"error": "O texto é demasiado longo (máx 120k caracteres). Tenta enviar por partes."}), 400

    try:
        # Chama a função principal
        resumo = chamar_perplexity(texto_tc, estilo)
        return jsonify({"summary": resumo})
    
    except Exception as e:
        # Log do erro no servidor (aparece nos logs do Render)
        print(f"Erro interno: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Este bloco só corre em desenvolvimento local.
    # No Render, o Gunicorn é usado e este bloco é ignorado.
    app.run(debug=True)
