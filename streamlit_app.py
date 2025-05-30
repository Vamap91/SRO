import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import json
import re

st.set_page_config(page_title="Analisador SRO", layout="wide")
st.title("🔍 Analisador SRO - Previsão de Reclamações")

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

# Função para aplicar a lógica do Prompt V2
def analisar_risco(texto):
    texto = texto.lower()
    pontos = 0

    # Frequência de contato (peso 4)
    if any(t in texto for t in ["ligou novamente", "cliente retornou", "novo contato", "reforcado"]):
        freq_nota = 10
    elif texto.count("contato") >= 2:
        freq_nota = 5
    else:
        freq_nota = 2
    pontos += freq_nota * 4

    # Tempo de espera (peso 3)
    atraso = any(t in texto for t in ["aguardando retorno", "sem previsao", "sem posicionamento", "prazo vencido", "reagendado"])
    pontos += (10 if atraso else 2) * 3

    # Falhas processuais (peso 2)
    falha = any(t in texto for t in ["modelo errado", "endereco divergente", "nao foi informado", "nao retornaram", "peca errada", "tecnico nao apareceu", "problema tecnico", "falha na execucao"])
    pontos += (10 if falha else 2) * 2

    # Estado emocional (peso 1)
    emocional = any(t in texto for t in ["irritado", "frustrado", "insatisfeito", "ameacou reclamar", "descontente", "cobrou solucao"])
    pontos += (10 if emocional else 2) * 1

    percentual = int(pontos)
    if percentual <= 30:
        classificacao = "Baixa"
    elif percentual <= 60:
        classificacao = "Media"
    elif percentual <= 85:
        classificacao = "Alta"
    else:
        classificacao = "Critica"

    return percentual, classificacao

# Upload de arquivo
uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        coluna = st.selectbox("Selecione a coluna com os comentarios:", df.columns)
        df = df[[coluna]].rename(columns={coluna: "Comentario"})

    elif uploaded_file.name.endswith(".pdf"):
        df = extract_text_from_pdf(uploaded_file)

    elif uploaded_file.name.endswith(".json"):
        df = extract_text_from_json(uploaded_file)

    # Aplica a análise
    df[["Porcentagem de Reclamação", "Classificacao"]] = df["Comentario"].apply(lambda x: pd.Series(analisar_risco(str(x))))

    st.success("Análise concluída com sucesso!")
    st.dataframe(df)

    # Download do Excel
    output = df.to_excel(index=False, engine='openpyxl')
    st.download_button("📂 Baixar Relatório Excel", data=output, file_name="relatorio_sro.xlsx")
