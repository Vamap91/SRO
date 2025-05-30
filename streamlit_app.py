import streamlit as st
import pandas as pd
import fitz # PyMuPDF
import json
import re
from openai import OpenAI
from io import BytesIO
# from fpdf import FPDF # Comentado, pois fpdf não está disponível no ambiente do Code Interpreter

st.set_page_config(page_title="Analisador SRO - Previsão de Reclamações")
st.title("🔍 Analisador SRO - Previsão de Reclamações")

# Configurar chave da OpenAI
# Certifique-se de que st.secrets["OPENAI_API_KEY"] está configurado corretamente
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Prompt V2 como system_message
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

# Função para extrair texto de PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    # Retorna cada linha como um comentário separado
    return pd.DataFrame({"Comentario": [line for line in text.split("\n") if line.strip()]})

# Função para extrair texto de JSON
def extract_text_from_json(uploaded_file):
    data = json.load(uploaded_file)
    comments = []
    if isinstance(data, list):
        for entry in data:
            comments.append(str(entry))
    elif isinstance(data, dict):
        for key, value in data.items():
            comments.append(f"{key}: {value}") # Inclui chave para contexto
    else:
        comments.append(str(data))
    return pd.DataFrame({"Comentario": comments})

# Função com chamada para OpenAI GPT
@st.cache_data(show_spinner=False)
def analisar_comentario_openai(comentario, historico_df):
    try:
        # Selecionar 5 comentários reais da base histórica (se houver)
        # Atenção: Esta é uma simulação para o ambiente de teste.
        # No seu ambiente real, 'historico_df' deve ser carregado de forma mais robusta.
        if not historico_df.empty:
            exemplos_reais = historico_df.sample(n=min(5, len(historico_df)), random_state=42)["Comentario"].tolist()
            contexto_exemplos = "Exemplos de comentários reais que geraram reclamação:\n- " + "\n- ".join(exemplos_reais)
        else:
            contexto_exemplos = "Não há exemplos históricos disponíveis para referência."

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

# A função gerar_pdf foi comentada devido à indisponibilidade de fpdf no ambiente.
# def gerar_pdf(df):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     risco_icon = {
#         "Baixa": "🟢",
#         "Média": "🟡",
#         "Alta": "🟠",
#         "Crítica": "🔴"
#     }

#     for idx, row in df.iterrows():
#         resultado = row["Resultado IA"]
#         risco = ""
#         for nivel in risco_icon:
#             if f"Probabilidade de Reclamação: {nivel}" in resultado:
#                 risco = risco_icon[nivel] + " " + nivel
#                 break

#         pdf.set_font("Arial", 'B', 11)
#         pdf.cell(0, 10, f"Comentário {idx + 1} - Risco: {risco}", ln=True)
#         pdf.set_font("Arial", '', 11)
#         for linha in resultado.split("\n"):
#             if not linha.startswith("- Pedido"):
#                 pdf.multi_cell(0, 8, linha.strip())
#         pdf.ln(4)
#         pdf.cell(190, 0, '', ln=True, border='T')
#         pdf.ln(6)

#     buffer = BytesIO()
#     buffer.write(pdf.output(dest='S').encode('latin1'))
#     buffer.seek(0)
#     return buffer

# --- Lógica Principal ---
# Carrega o histórico de comentários do arquivo CSV fornecido pelo usuário
try:
    # Tentativa de carregar o arquivo excel como csv (assumindo que "Informações SRO.xlsx - Planilha3.csv" é o arquivo relevante)
    historico_df_path = "Informações SRO.xlsx - Planilha3.csv"
    historico_df_raw = pd.read_csv(historico_df_path) # Use pd.read_csv para CSV
    # Renomear a Coluna A para "Comentario" conforme discutido
    # Assumindo que a primeira coluna é a "Coluna A"
    historico_df = historico_df_raw.iloc[:, 0].to_frame(name="Comentario")
    historico_df = historico_df.dropna().drop_duplicates().reset_index(drop=True)
except Exception as e:
    st.warning(f"Não foi possível carregar a base histórica 'Informações SRO.xlsx - Planilha3.csv'. A análise será feita sem exemplos históricos. Erro: {e}")
    historico_df = pd.DataFrame({"Comentario": []}) # DataFrame vazio se houver erro

uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json"])

df = pd.DataFrame() # Inicializa df fora do if para evitar NameError

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            if df.shape[0] == 1 and df.shape[1] > 1:
                # Se for uma única linha com múltiplos campos, concatena
                comentario = ' '.join([str(v) for v in df.iloc[0].values if pd.notna(v)])
                df = pd.DataFrame({"Pedido": ["Pedido 1"], "Comentario": [comentario]})
            elif df.shape[1] == 1:
                # Se for uma única coluna, assume que é o comentário
                df.insert(0, "Pedido", [f"Linha {i+1}" for i in range(len(df))])
                df.columns = ["Pedido", "Comentario"]
            else:
                # Tenta identificar colunas de pedido e comentário
                colunas_disponiveis = df.columns.tolist()
                st.write(f"Colunas disponíveis: {colunas_disponiveis}")

                # Tentativa de identificar colunas "Pedido" e "Comentario" automaticamente
                coluna_pedido = None
                coluna_comentario = None

                for col in colunas_disponiveis:
                    if "pedido" in col.lower() or "os" in col.lower() or "id" in col.lower():
                        if coluna_pedido is None: # Prioriza a primeira encontrada
                            coluna_pedido = col
                    if "comentario" in col.lower() or "anotacao" in col.lower() or "actionresult" in col.lower():
                        if coluna_comentario is None: # Prioriza a primeira encontrada
                            coluna_comentario = col

                if coluna_pedido and coluna_comentario and coluna_pedido != coluna_comentario:
                    st.success(f"Colunas identificadas automaticamente: Pedido='{coluna_pedido}', Comentário='{coluna_comentario}'")
                    df = df[[coluna_pedido, coluna_comentario]].rename(columns={coluna_pedido: "Pedido", coluna_comentario: "Comentario"})
                    df["Comentario"] = df.groupby("Pedido")["Comentario"].transform(lambda x: '\n'.join(x.astype(str)))
                    df = df.drop_duplicates(subset=["Pedido"]).reset_index(drop=True)
                else:
                    # Se não identificar automaticamente, pede para o usuário selecionar
                    st.warning("Não foi possível identificar as colunas de 'Pedido' e 'Comentário' automaticamente. Por favor, selecione-as manualmente.")
                    colunas_selecionadas = st.multiselect(
                        "Selecione as colunas (primeira deve ser o Pedido/ID, segunda os Comentários):",
                        df.columns,
                        default=colunas_disponiveis[:2] if len(colunas_disponiveis) >= 2 else []
                    )
                    if len(colunas_selecionadas) >= 2:
                        df = df[colunas_selecionadas[:2]]
                        df.columns = ["Pedido", "Comentario"]
                        df["Comentario"] = df.groupby("Pedido")["Comentario"].transform(lambda x: '\n'.join(x.astype(str)))
                        df = df.drop_duplicates(subset=["Pedido"]).reset_index(drop=True)
                    else:
                        st.error("Selecione pelo menos duas colunas: uma para o número do pedido/ID e outra para os comentários.")
                        st.stop()
        except Exception as e:
            st.error(f"Erro ao ler o arquivo Excel: {e}")
            st.stop()

    elif uploaded_file.name.endswith(".pdf"):
        df = extract_text_from_pdf(uploaded_file)
        # Adiciona uma coluna de "Pedido" fictícia para PDFs
        df.insert(0, "Pedido", [f"PDF-{i+1}" for i in range(len(df))])

    elif uploaded_file.name.endswith(".json"):
        df = extract_text_from_json(uploaded_file)
        # Adiciona uma coluna de "Pedido" fictícia para JSONs
        df.insert(0, "Pedido", [f"JSON-{i+1}" for i in range(len(df))])

    if not df.empty:
        with st.spinner("Analisando os pedidos com IA..."):
            # Passa o DataFrame histórico para a função de análise
            df["Resultado IA"] = df["Comentario"].apply(lambda x: analisar_comentario_openai(str(x), historico_df))

        st.success("Análise concluída com sucesso!")
        st.dataframe(df)

        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        st.download_button("📂 Baixar Relatório Excel", data=output, file_name="relatorio_sro.xlsx")

        # pdf_buffer = gerar_pdf(df) # Comentado, pois fpdf não está disponível no ambiente.
        # st.download_button("📝 Baixar Relatório PDF", data=pdf_buffer, file_name="relatorio_sro.pdf")
