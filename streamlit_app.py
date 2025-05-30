import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import json
import re
from openai import OpenAI
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Analisador SRO - Previsão de Reclamações")
st.title("🔍 Analisador SRO - Previsão de Reclamações")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SRO_PROMPT_V2 = """
Você é um especialista em análise preditiva de qualidade para uma empresa de serviços automotivos especializada em troca e reparo de vidros (VFLR) e funilaria/martelinho de ouro (RRSM). Seu papel é avaliar a chance de uma ordem de serviço gerar uma reclamação, com base exclusivamente em comentários deixados por atendentes após ligações com clientes.

Considere como sinais de risco os seguintes padrões frequentemente observados em reclamações reais:

- Palavras como: “informa”, “contato”, “retorno”, “troca”, “peça”, “ciente”, “segurado”, “corretor(a)”
- Repetição de contato ou falha de comunicação
- Atraso no atendimento ou ausência de follow-up
- Problemas técnicos ou execução inadequada do serviço
- Expressões emocionais negativas ou frustração

Dada uma nova anotação de atendimento, responda com:
- Pedido: N/A
- Probabilidade de Reclamação: Baixa / Média / Alta / Crítica
- Porcentagem de Reclamação: XX%
- Fatores Críticos: Explique quais sinais do texto contribuíram para o risco
- Conclusão: Resuma o risco e sugira ações se for médio ou superior
"""

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return pd.DataFrame({"Comentario": text.split("\n")})

def extract_text_from_json(uploaded_file):
    data = json.load(uploaded_file)
    if isinstance(data, list):
        return pd.DataFrame({"Comentario": [str(entry) for entry in data]})
    elif isinstance(data, dict):
        return pd.DataFrame({"Comentario": [str(entry) for entry in data.values()]})
    else:
        return pd.DataFrame({"Comentario": [str(data)]})

@st.cache_data(show_spinner=False)
def analisar_comentario_openai(comentario):
    try:
        # Carregar base histórica real de comentários
        historico_path = "Informações SRO.xlsx"
        try:
            historico_df = pd.read_excel(historico_path)
            historico_df = historico_df.rename(columns={"ActionResult": "Comentario"})
            historico_df = historico_df[["Comentario"]].dropna().drop_duplicates().reset_index(drop=True)
        except:
            historico_df = pd.DataFrame({"Comentario": [
                "Serviço mal feito, cliente insatisfeito.",
                "Cliente voltou com problema no vidro mal instalado.",
                "Falta de retorno gerou frustração do cliente.",
                "Erro na cor da pintura gerou nova visita.",
                "Cliente aguardou mais de 2 horas sem solução."
            ]})

        # Selecionar 5 comentários reais da base
        exemplos_reais = historico_df.sample(n=min(5, len(historico_df)), random_state=42)["Comentario"].tolist()
        contexto_exemplos = "Exemplos de comentários reais que geraram reclamação:"
" + "
- " + "
- ".join(exemplos_reais)
- " + "
- ".join(exemplos_reais)

        prompt_usuario = f"""
{contexto_exemplos}

Agora analise o novo comentário:
"{comentario}"

Dê o resultado no seguinte formato:
- Pedido: N/A
- Probabilidade de Reclamação: Baixa / Média / Alta / Crítica
- Porcentagem de Reclamação: XX%
- Fatores Críticos: Explique quais sinais do texto contribuíram para o risco
- Conclusão: Resuma o risco e sugira ações se for médio ou superior
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SRO_PROMPT_V2},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise: {str(e)}"
    except Exception as e:
        return f"Erro na análise: {str(e)}"
    except Exception as e:
        return f"Erro na análise: {str(e)}"

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
        for linha in resultado.split("\n"):
            if not linha.startswith("- Pedido"):
                pdf.multi_cell(0, 8, linha.strip())
        pdf.ln(4)
        pdf.cell(190, 0, '', ln=True, border='T')
        pdf.ln(6)

    buffer = BytesIO()
    buffer.write(pdf.output(dest='S').encode('latin1'))
    buffer.seek(0)
    return buffer

uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            if df.shape[0] == 1 and df.shape[1] > 1:
                comentario = ' '.join([str(v) for v in df.iloc[0].values if pd.notna(v)])
                df = pd.DataFrame({"Pedido": ["Pedido 1"], "Comentario": [comentario]})
            elif df.shape[1] == 1:
                df.insert(0, "Pedido", [f"Linha {i+1}" for i in range(len(df))])
                df.columns = ["Pedido", "Comentario"]
            else:
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

    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button("📂 Baixar Relatório Excel", data=output, file_name="relatorio_sro.xlsx")

    pdf_buffer = gerar_pdf(df)
    st.download_button("📝 Baixar Relatório PDF", data=pdf_buffer, file_name="relatorio_sro.pdf")
