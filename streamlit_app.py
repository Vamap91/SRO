import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import fitz  # PyMuPDF
import openai
from io import BytesIO
import faiss
import time
import base64
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="SRO - Previsão de Reclamações",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .risk-low {
        color: green;
        font-weight: bold;
    }
    .risk-medium {
        color: orange;
        font-weight: bold;
    }
    .risk-high {
        color: red;
        font-weight: bold;
    }
    .risk-critical {
        color: darkred;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% {
            opacity: 0.7;
        }
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo {
        height: 60px;
        margin-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho da aplicação
st.markdown("""
<div class="header-container">
    <div class="logo-container">
        <h1>📊 SRO - Sistema de Previsão de Reclamações</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# Inicialização silenciosa da chave da API OpenAI diretamente dos secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except Exception:
    # Tratamento silencioso - não exibe mensagem de erro na inicialização
    pass

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Configurações do modelo
    st.subheader("Modelo de IA")
    model = st.selectbox(
        "Selecione o modelo:",
        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    # Configurações RAG
    st.subheader("Configurações RAG")
    num_exemplos = st.slider("Número de exemplos históricos:", 1, 10, 5)
    
    # Informações sobre o projeto
    st.markdown("---")
    st.markdown("### Sobre o Projeto SRO")
    st.markdown("""
    Este aplicativo utiliza Inteligência Artificial com RAG (Retrieval Augmented Generation) 
    para prever a probabilidade de reclamações com base em comentários de atendimento.
    """)

# Função para extrair texto de PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Tenta dividir o texto em parágrafos ou seções
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            df = pd.DataFrame({"Comentario": [p.strip() for p in paragraphs if p.strip()]})
        else:
            df = pd.DataFrame({"Comentario": [text]})
        return df
    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
        return pd.DataFrame({"Comentario": []})

# Função para extrair texto de JSON
def extract_text_from_json(uploaded_file):
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        
        # Tenta encontrar comentários no JSON
        comments = []
        
        # Função recursiva para extrair valores de texto de um JSON aninhado
        def extract_text_values(obj, key_pattern="coment"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and key_pattern.lower() in k.lower():
                        comments.append(v)
                    elif isinstance(v, (dict, list)):
                        extract_text_values(v, key_pattern)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_values(item, key_pattern)
        
        extract_text_values(data)
        
        if not comments:
            # Se não encontrou comentários específicos, pega todos os valores de string
            def extract_all_strings(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, str) and len(v) > 10:  # Strings com mais de 10 caracteres
                            comments.append(v)
                        elif isinstance(v, (dict, list)):
                            extract_all_strings(v)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_all_strings(item)
            
            extract_all_strings(data)
        
        return pd.DataFrame({"Comentario": comments})
    except Exception as e:
        st.error(f"Erro ao processar o JSON: {e}")
        return pd.DataFrame({"Comentario": []})

# Função para gerar embeddings usando a API da OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    if not text or not isinstance(text, str):
        return np.zeros(1536)  # Retorna um vetor de zeros se o texto for inválido
    
    try:
        text = text.replace("\n", " ")
        response = openai.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(1536)  # Retorna um vetor de zeros em caso de erro

# Função para carregar e indexar a base histórica
@st.cache_resource
def load_historical_data(file_path):
    try:
        # Carregar o arquivo Excel
        df = pd.read_excel(file_path)
        
        # Verificar se há uma coluna de comentários
        comment_col = None
        for col in df.columns:
            if "coment" in col.lower() or "anota" in col.lower():
                comment_col = col
                break
        
        if not comment_col:
            comment_col = df.columns[0]  # Usa a primeira coluna se não encontrar uma específica
        
        # Filtrar linhas com comentários válidos
        df = df[[comment_col]].rename(columns={comment_col: "Comentario"})
        df = df.dropna(subset=["Comentario"]).reset_index(drop=True)
        df = df[df["Comentario"].astype(str).str.len() > 5]  # Filtra comentários muito curtos
        
        # Gerar embeddings para cada comentário
        st.info(f"Gerando embeddings para {len(df)} comentários históricos. Isso pode levar alguns minutos...")
        
        # Processar em lotes para evitar sobrecarga da API
        batch_size = 100
        all_embeddings = []
        
        progress_bar = st.progress(0)
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_embeddings = [get_embedding(text) for text in batch["Comentario"].astype(str)]
            all_embeddings.extend(batch_embeddings)
            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
        
        progress_bar.empty()
        
        # Converter para array numpy
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Criar índice FAISS
        dimension = embeddings_array.shape[1]  # Dimensão dos embeddings
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        return df, index, embeddings_array
    
    except Exception as e:
        st.error(f"Erro ao carregar dados históricos: {e}")
        return pd.DataFrame({"Comentario": []}), None, None

# Função para buscar comentários similares
def find_similar_comments(query_embedding, index, df, embeddings_array, k=5):
    if index is None or embeddings_array is None:
        return pd.DataFrame({"Comentario": []})
    
    try:
        # Buscar os k comentários mais similares
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Criar DataFrame com os resultados
        similar_df = pd.DataFrame({
            "Comentario": df.iloc[indices[0]]["Comentario"].values,
            "Similaridade": 1 - (distances[0] / np.max(distances[0]) if np.max(distances[0]) > 0 else distances[0])
        })
        
        return similar_df
    
    except Exception as e:
        st.error(f"Erro na busca de comentários similares: {e}")
        return pd.DataFrame({"Comentario": []})

# Função para analisar comentário com GPT-4
def analisar_comentario_openai(comentario, similar_comments, model="gpt-4"):
    try:
        # Construir o prompt com os exemplos históricos
        exemplos_historicos = ""
        for i, (_, row) in enumerate(similar_comments.iterrows(), 1):
            exemplos_historicos += f"Exemplo {i} (Similaridade: {row['Similaridade']:.2f}):\n{row['Comentario']}\n\n"
        
        system_message = """Você é um especialista em análise preditiva de qualidade de atendimento ao cliente. 
Sua tarefa é analisar comentários de atendimento e prever a probabilidade de uma reclamação formal ser aberta.

Considere os seguintes fatores e seus pesos para sua análise:

1. Frequência de Contatos (Peso 4):
   - 1 contato: baixo risco
   - 2 contatos: médio risco
   - 3+ contatos: risco elevado

2. Tempo de Espera (Peso 3):
   - Negociação Carglass: > 1 dia útil
   - Acompanhamento de peças (VFLR): > 5 dias úteis
   - Agendamento: > 1 dia útil
   - Confirmação de execução: qualquer atraso

3. Falhas Processuais (Peso 2):
   - Cadastro incorreto (endereço/placa/modelo)
   - Solicitações específicas não atendidas
   - Falhas de comunicação entre setores
   - Problemas técnicos após execução do serviço (gravidade alta)

4. Estado Emocional do Cliente (Peso 1):
   - Indicações de frustração, irritação, insatisfação

Classifique o risco e a porcentagem:
- Baixa: 0-30%
- Média: 31-60%
- Alta: 61-85%
- Crítica: 86-100%

Sua resposta deve seguir EXATAMENTE este formato:
Probabilidade de Reclamação: [Baixa/Média/Alta/Crítica]
Porcentagem de Reclamação: [XX%]
Fatores Críticos: [Liste os fatores que contribuíram para o risco]
Conclusão: [Resumo conciso da análise com sugestão de ação preventiva se o risco for Médio ou superior]
"""

        user_message = f"""Analise o seguinte comentário de atendimento e determine a probabilidade de uma reclamação formal ser aberta:

COMENTÁRIO A ANALISAR:
{comentario}

COMENTÁRIOS HISTÓRICOS SIMILARES (todos estes casos resultaram em reclamações formais):
{exemplos_historicos}

Com base no comentário e nos exemplos históricos similares, determine a probabilidade de uma reclamação formal ser aberta.
"""

        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Erro na análise: {e}"

# Função para extrair informações da resposta da IA
def extract_info_from_ai_response(response):
    try:
        lines = response.strip().split('\n')
        info = {}
        
        for line in lines:
            if "Probabilidade de Reclamação:" in line:
                info["probabilidade"] = line.split(":", 1)[1].strip()
            elif "Porcentagem de Reclamação:" in line:
                percentage_text = line.split(":", 1)[1].strip()
                percentage = ''.join(filter(lambda x: x.isdigit() or x == '.', percentage_text))
                info["porcentagem"] = float(percentage)
            elif "Fatores Críticos:" in line:
                info["fatores"] = line.split(":", 1)[1].strip()
            elif "Conclusão:" in line:
                info["conclusao"] = line.split(":", 1)[1].strip()
        
        return info
    except Exception as e:
        st.error(f"Erro ao extrair informações da resposta da IA: {e}")
        return {
            "probabilidade": "Erro",
            "porcentagem": 0,
            "fatores": "Erro na extração",
            "conclusao": "Não foi possível analisar a resposta da IA."
        }

# Função para formatar a exibição do resultado
def format_result_display(pedido, ai_response):
    info = extract_info_from_ai_response(ai_response)
    
    probabilidade = info.get("probabilidade", "Erro")
    porcentagem = info.get("porcentagem", 0)
    fatores = info.get("fatores", "Não identificados")
    conclusao = info.get("conclusao", "Não disponível")
    
    risk_class = ""
    if "baixa" in probabilidade.lower():
        risk_class = "risk-low"
    elif "média" in probabilidade.lower():
        risk_class = "risk-medium"
    elif "alta" in probabilidade.lower():
        risk_class = "risk-high"
    elif "crítica" in probabilidade.lower():
        risk_class = "risk-critical"
    
    html = f"""
    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
        <h3>Pedido: {pedido}</h3>
        <p><strong>Probabilidade de Reclamação:</strong> <span class="{risk_class}">{probabilidade}</span></p>
        <p><strong>Porcentagem de Risco:</strong> <span class="{risk_class}">{porcentagem}%</span></p>
        <p><strong>Fatores Críticos:</strong> {fatores}</p>
        <p><strong>Conclusão:</strong> {conclusao}</p>
    </div>
    """
    return html

# Função principal
def main():
    # Carregar dados históricos
    file_path = "InformaçõesSRO.xlsx"
    
    if os.path.exists(file_path):
        with st.spinner("Carregando e indexando dados históricos..."):
            historico_df, index, embeddings_array = load_historical_data(file_path)
            if not historico_df.empty and index is not None:
                st.success(f"✅ Base histórica carregada com {len(historico_df)} registros")
            else:
                st.error("❌ Erro ao carregar a base histórica")
    else:
        st.error(f"❌ Arquivo {file_path} não encontrado. A análise será feita sem exemplos históricos.")
        historico_df = pd.DataFrame({"Comentario": []})
        index = None
        embeddings_array = None
    
    # Upload de arquivo
    st.subheader("📤 Upload de Arquivo")
    uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json"])
    
    # Inicializa DataFrame fora do if para evitar NameError
    df = pd.DataFrame()
    
    if uploaded_file:
        with st.spinner("Processando arquivo..."):
            if uploaded_file.name.endswith(".xlsx"):
                try:
                    df = pd.read_excel(uploaded_file)
                    
                    # Exibir informações sobre o arquivo
                    st.info(f"Arquivo carregado: {uploaded_file.name} | {df.shape[0]} linhas x {df.shape[1]} colunas")
                    
                    # Se for uma única linha com múltiplos campos, concatena
                    if df.shape[0] == 1 and df.shape[1] > 1:
                        comentario = ' '.join([str(v) for v in df.iloc[0].values if pd.notna(v)])
                        df = pd.DataFrame({"Pedido": ["Pedido 1"], "Comentario": [comentario]})
                    elif df.shape[1] == 1:
                        # Se for uma única coluna, assume que é o comentário
                        df.insert(0, "Pedido", [f"Linha {i+1}" for i in range(len(df))])
                        df.columns = ["Pedido", "Comentario"]
                    else:
                        # Tenta identificar colunas de pedido e comentário
                        colunas_disponiveis = df.columns.tolist()
                        
                        # Tentativa de identificar colunas "Pedido" e "Comentario" automaticamente
                        coluna_pedido = None
                        coluna_comentario = None
                        
                        for col in colunas_disponiveis:
                            col_lower = col.lower()
                            if "pedido" in col_lower or "os" in col_lower or "id" in col_lower or "protocolo" in col_lower:
                                if coluna_pedido is None:  # Prioriza a primeira encontrada
                                    coluna_pedido = col
                            if "comentario" in col_lower or "anotacao" in col_lower or "obs" in col_lower or "descricao" in col_lower:
                                if coluna_comentario is None:  # Prioriza a primeira encontrada
                                    coluna_comentario = col
                        
                        if coluna_pedido and coluna_comentario and coluna_pedido != coluna_comentario:
                            st.success(f"✅ Colunas identificadas automaticamente: Pedido='{coluna_pedido}', Comentário='{coluna_comentario}'")
                            df = df[[coluna_pedido, coluna_comentario]].rename(columns={coluna_pedido: "Pedido", coluna_comentario: "Comentario"})
                            df["Comentario"] = df.groupby("Pedido")["Comentario"].transform(lambda x: '\n'.join(x.astype(str)))
                            df = df.drop_duplicates(subset=["Pedido"]).reset_index(drop=True)
                        else:
                            # Se não identificar automaticamente, pede para o usuário selecionar
                            st.warning("⚠️ Não foi possível identificar as colunas de 'Pedido' e 'Comentário' automaticamente. Por favor, selecione-as manualmente.")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                coluna_pedido = st.selectbox(
                                    "Selecione a coluna de Pedido/ID:",
                                    options=colunas_disponiveis,
                                    index=0 if colunas_disponiveis else None
                                )
                            
                            with col2:
                                coluna_comentario = st.selectbox(
                                    "Selecione a coluna de Comentários:",
                                    options=colunas_disponiveis,
                                    index=min(1, len(colunas_disponiveis)-1) if len(colunas_disponiveis) > 1 else None
                                )
                            
                            if coluna_pedido and coluna_comentario:
                                df = df[[coluna_pedido, coluna_comentario]].rename(columns={coluna_pedido: "Pedido", coluna_comentario: "Comentario"})
                                df["Comentario"] = df.groupby("Pedido")["Comentario"].transform(lambda x: '\n'.join(x.astype(str)))
                                df = df.drop_duplicates(subset=["Pedido"]).reset_index(drop=True)
                            else:
                                st.error("❌ Selecione as colunas de Pedido e Comentário para continuar.")
                                st.stop()
                
                except Exception as e:
                    st.error(f"❌ Erro ao ler o arquivo Excel: {e}")
                    st.stop()
            
            elif uploaded_file.name.endswith(".pdf"):
                df = extract_text_from_pdf(uploaded_file)
                # Adiciona uma coluna de "Pedido" fictícia para PDFs
                df.insert(0, "Pedido", [f"PDF-{i+1}" for i in range(len(df))])
            
            elif uploaded_file.name.endswith(".json"):
                df = extract_text_from_json(uploaded_file)
                # Adiciona uma coluna de "Pedido" fictícia para JSONs
                df.insert(0, "Pedido", [f"JSON-{i+1}" for i in range(len(df))])
        
        # Processar os comentários se o DataFrame não estiver vazio
        if not df.empty:
            # Exibir prévia dos dados
            st.subheader("📋 Prévia dos Dados")
            st.dataframe(df.head(5))
            
            # Botão para iniciar análise
            if st.button("🔍 Analisar Comentários"):
                # Verificar se a API OpenAI está configurada
                if not openai.api_key:
                    st.error("❌ Chave da API OpenAI não configurada. Verifique as configurações do Streamlit.")
                    st.stop()
                
                # Adicionar coluna para resultados
                df["Resultado IA"] = ""
                df["HTML"] = ""
                
                # Processar cada comentário
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, row in enumerate(df.iterrows()):
                    idx, data = row
                    pedido = data["Pedido"]
                    comentario = data["Comentario"]
                    
                    status_text.text(f"Analisando pedido {pedido} ({i+1}/{len(df)})...")
                    
                    # Gerar embedding para o comentário atual
                    comentario_embedding = get_embedding(str(comentario))
                    
                    # Buscar comentários similares na base histórica
                    similar_comments = find_similar_comments(
                        comentario_embedding, 
                        index, 
                        historico_df, 
                        embeddings_array, 
                        k=num_exemplos
                    )
                    
                    # Analisar com GPT-4
                    resultado = analisar_comentario_openai(str(comentario), similar_comments, model=model)
                    df.at[idx, "Resultado IA"] = resultado
                    
                    # Formatar HTML para exibição
                    df.at[idx, "HTML"] = format_result_display(pedido, resultado)
                    
                    # Atualizar barra de progresso
                    progress_bar.progress((i + 1) / len(df))
                    
                    # Pequena pausa para evitar rate limits da API
                    time.sleep(0.1)
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Análise concluída com sucesso!")
                
                # Exibir resultados
                st.subheader("📊 Resultados da Análise")
                
                # Exibir cada resultado formatado em HTML
                for _, row in df.iterrows():
                    st.markdown(row["HTML"], unsafe_allow_html=True)
                
                # Opções de download
                st.subheader("📥 Download dos Resultados")
                
                # Preparar Excel para download
                output_excel = BytesIO()
                df_download = df.drop(columns=["HTML"])
                df_download.to_excel(output_excel, index=False, engine='openpyxl')
                output_excel.seek(0)
                
                # Botão de download Excel
                st.download_button(
                    "📊 Baixar Relatório Excel",
                    data=output_excel,
                    file_name=f"relatorio_sro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
