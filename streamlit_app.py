import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import json
import re
from openai import OpenAI
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Analisador SRO", layout="wide")
st.title("🔍 Analisador SRO - Previsão de Reclamações")

# Instanciar cliente OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Prompt V2 como system_message
SRO_PROMPT_V2 = """
Você é um especialista em análise preditiva de qualidade para uma empresa de serviços automotivos especializada em troca e reparo de vidros (VFLR) e funilaria/martelinho de ouro (RRSM). Seu papel é avaliar a chance de uma ordem de serviço gerar uma reclamação, com base exclusivamente em comentários deixados por atendentes após ligações com clientes.

Atenção: todos os comentários do dataset de treinamento representam casos reais onde houve abertura de não conformidade (reclamação). Isso significa que o seu trabalho é identificar, entre novos comentários, quais se assemelham ou seguem padrões perigosos observados nesse histórico.

Dada uma nova anotação de atendimento, responda com:
- Pedido: N/A
- Probabilidade de Reclamação: Baixa / Média / Alta / Crítica
- Porcentagem de Reclamação: XX%
- Fatores Críticos: Explique quais sinais do texto contribuíram para o risco
- Conclusão: Resuma o risco e sugira ações se for médio ou superior
"""

# Função para extrair texto de PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return pd.DataFrame({"Comentario": text.split("\n")})

# Função para extrair texto de JSON
def extract_text_from_json(uploaded_file):
    data = json.load(uploaded_file)
    if isinstance(data, list):
        return pd.DataFrame({"Comentario": [str(entry) for entry in data]})
    elif isinstance(data, dict):
        return pd.DataFrame({"Comentario": [str(entry) for entry in data.values()]})
    else:
        return pd.DataFrame({"Comentario": [str(data)]})

# Função com chamada para OpenAI GPT
@st.cache_data(show_spinner=False)
def analisar_comentario_openai(comentario):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SRO_PROMPT_V2},
                {"role": "user", "content": comentario}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise: {str(e)}"

# Geração de PDF visual com ícones
def gerar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    risco_icon = {
        "Baixa": "🟢",
        "Média": "🟡",
        "Alta": "🟠",
        "Crítica": "🔴"
    }

    for idx, row in df.iterrows():
        resultado = row["Resultado IA"]

        risco = ""
        for nivel in risco_icon:
            if f"Probabilidade de Reclamação: {nivel}" in resultado:
                risco = risco_icon[nivel] + " " + nivel
                break

        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f"Comentário {idx + 1} - Risco: {risco}", ln=True)
        pdf.set_font("Arial", '', 11)
        linhas = resultado.split("\n")
        for linha in linhas:
            if not linha.startswith("- Pedido"):
                pdf.multi_cell(0, 8, linha.strip())
        pdf.ln(4)
        pdf.cell(190, 0, '', ln=True, border='T')
        pdf.ln(6)

    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer = BytesIO()
    buffer.write(pdf_output)
    buffer.seek(0)
    return buffer

# Upload de arquivo
uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            colunas = st.multiselect("Selecione as colunas:", df.columns, default=df.columns[:2])
            if len(colunas) >= 2:
                df = df[colunas[:2]]
                df.columns = ["Pedido", "Comentario"]
                df = df.groupby("Pedido")["Comentario"].apply(lambda x: '\n'.join(x)).reset_index()
            else:
                st.error("Selecione pelo menos duas colunas: número do pedido e comentários.")
                st.stop()
        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")
            st.stop()

    elif uploaded_file.name.endswith(".pdf"):
        df = extract_text_from_pdf(uploaded_file)
        df.insert(0, "Pedido", [f"PDF-{i}" for i in range(len(df))])

    elif uploaded_file.name.endswith(".json"):
        df = extract_text_from_json(uploaded_file)
        df.insert(0, "Pedido", [f"JSON-{i}" for i in range(len(df))])

    with st.spinner("Analisando os pedidos com IA..."):
        df["Resultado IA"] = df["Comentario"].apply(lambda x: analisar_comentario_openai(str(x)))

    st.success("Análise concluída com sucesso!")
    st.dataframe(df)

    # Download do Excel corrigido
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button("📂 Baixar Relatório Excel", data=output, file_name="relatorio_sro.xlsx")

    # Download do PDF organizado
    pdf_buffer = gerar_pdf(df)
    st.download_button("📝 Baixar Relatório PDF", data=pdf_buffer, file_name="relatorio_sro.pdf")
