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

# Geração de PDF limpo
def gerar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for idx, row in df.iterrows():
        resultado = row["Resultado IA"]
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, f"{resultado}")
        pdf.ln(5)
        pdf.cell(190, 0, '', ln=True, border='T')
        pdf.ln(5)

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
            if df.empty or df.shape[1] == 1 and df.columns[0].startswith("Unnamed"):
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, header=None)
                df = df.iloc[:, [0]]
                df.columns = ["Comentario"]
            else:
                coluna = st.selectbox("Selecione a coluna com os comentários:", df.columns)
                df = df[[coluna]].rename(columns={coluna: "Comentario"})
        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")
            st.stop()

    elif uploaded_file.name.endswith(".pdf"):
        df = extract_text_from_pdf(uploaded_file)

    elif uploaded_file.name.endswith(".json"):
        df = extract_text_from_json(uploaded_file)

    with st.spinner("Analisando os comentários com IA..."):
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
