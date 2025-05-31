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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import csv

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
    .progress-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
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
    openai.api_key = st.secrets["OPENAI_API_KEY"]
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
    num_exemplos = st.slider("Número de exemplos históricos:", 3, 10, 5)
    
    # Informações sobre o projeto
    st.markdown("---")
    st.markdown("### Sobre o Projeto SRO")
    st.markdown("""
    Este aplicativo utiliza Inteligência Artificial com RAG (Retrieval Augmented Generation) 
    para prever a probabilidade de reclamações com base em comentários de atendimento.
    """)

# Constantes e configurações
FAISS_INDEX_PATH = "sro_faiss_index.bin"
CHUNKS_METADATA_PATH = "sro_chunks_metadata.csv"
HISTORICAL_DATA_PATH = "InformaçõesSRO.xlsx - Planila3.csv"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

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

# Função para dividir texto em chunks
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text or not isinstance(text, str):
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

# Função para salvar o índice FAISS
def save_faiss_index(index, file_path):
    try:
        faiss.write_index(index, file_path)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar índice FAISS: {e}")
        return False

# Função para carregar o índice FAISS
def load_faiss_index(file_path):
    try:
        if os.path.exists(file_path):
            index = faiss.read_index(file_path)
            return index
        return None
    except Exception as e:
        st.error(f"Erro ao carregar índice FAISS: {e}")
        return None

# Função para salvar metadados dos chunks
def save_chunks_metadata(chunks_metadata, file_path):
    try:
        # Remover a coluna de embedding antes de salvar para economizar espaço
        if 'embedding' in chunks_metadata.columns:
            chunks_metadata = chunks_metadata.drop(columns=['embedding'])
        
        chunks_metadata.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar metadados dos chunks: {e}")
        return False

# Função para carregar metadados dos chunks
def load_chunks_metadata(file_path):
    try:
        if os.path.exists(file_path):
            chunks_metadata = pd.read_csv(file_path)
            return chunks_metadata
        return None
    except Exception as e:
        st.error(f"Erro ao carregar metadados dos chunks: {e}")
        return None

# Função para carregar e indexar a base histórica com chunking
@st.cache_resource
def load_and_index_historical_data_with_chunking():
    # Verificar se os arquivos de índice e metadados já existem
    index = load_faiss_index(FAISS_INDEX_PATH)
    chunks_metadata = load_chunks_metadata(CHUNKS_METADATA_PATH)
    
    if index is not None and chunks_metadata is not None:
        st.success("✅ Base histórica carregada do cache!")
        return index, chunks_metadata
    
    # Se não existirem, processar a base histórica
    st.info("🔄 Processando base histórica pela primeira vez. Isso pode levar alguns minutos...")
    
    try:
        # Carregar o arquivo CSV
        if not os.path.exists(HISTORICAL_DATA_PATH):
            st.error(f"❌ Arquivo {HISTORICAL_DATA_PATH} não encontrado.")
            return None, None
        
        df = pd.read_csv(HISTORICAL_DATA_PATH)
        
        # Assumir que a primeira coluna (índice 0) contém os comentários
        comment_col = df.columns[0]
        
        # Filtrar linhas com comentários válidos
        df = df[[comment_col]].rename(columns={comment_col: "Comentario"})
        df = df.dropna(subset=["Comentario"]).reset_index(drop=True)
        df = df[df["Comentario"].astype(str).str.len() > 5]  # Filtra comentários muito curtos
        
        # Adicionar coluna de ID para rastreabilidade
        df["ID_Original"] = [f"OS_Hist_{i+1}" for i in range(len(df))]
        
        # Dividir comentários em chunks
        all_chunks = []
        all_embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_comments = len(df)
        
        for i, row in enumerate(df.iterrows()):
            idx, data = row
            comment = data["Comentario"]
            id_original = data["ID_Original"]
            
            status_text.text(f"Processando comentário {i+1}/{total_comments}...")
            
            # Dividir o comentário em chunks
            chunks = split_text_into_chunks(str(comment))
            
            # Para cada chunk, criar um registro com metadados
            for j, chunk in enumerate(chunks):
                chunk_id = f"{id_original}_chunk_{j+1}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                    "id_original": id_original,
                    "comentario_original": comment
                })
            
            # Atualizar barra de progresso
            progress_bar.progress((i + 1) / total_comments)
        
        # Criar DataFrame de chunks
        chunks_df = pd.DataFrame(all_chunks)
        
        # Gerar embeddings para cada chunk
        status_text.text("Gerando embeddings para chunks...")
        progress_bar.progress(0)
        
        total_chunks = len(chunks_df)
        batch_size = 100  # Processar em lotes para evitar sobrecarga da API
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks_df.iloc[i:min(i+batch_size, total_chunks)]
            
            batch_embeddings = []
            for _, row in batch.iterrows():
                embedding = get_embedding(row["chunk_text"])
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            # Atualizar barra de progresso
            progress_bar.progress(min(1.0, (i + batch_size) / total_chunks))
        
        # Adicionar embeddings ao DataFrame
        chunks_df["embedding"] = all_embeddings
        
        # Criar índice FAISS
        status_text.text("Criando índice FAISS...")
        
        # Converter para array numpy
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Criar índice
        dimension = embeddings_array.shape[1]  # Dimensão dos embeddings
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Salvar índice e metadados
        status_text.text("Salvando índice e metadados...")
        save_faiss_index(index, FAISS_INDEX_PATH)
        save_chunks_metadata(chunks_df, CHUNKS_METADATA_PATH)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Base histórica processada com sucesso! {len(chunks_df)} chunks gerados de {total_comments} comentários.")
        
        return index, chunks_df
    
    except Exception as e:
        st.error(f"❌ Erro ao processar base histórica: {e}")
        return None, None

# Função para buscar chunks similares
def retrieve_similar_examples(query_embedding, index, chunks_metadata, k=5):
    if index is None or chunks_metadata is None:
        return pd.DataFrame()
    
    try:
        # Buscar os k chunks mais similares
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Obter os chunks correspondentes
        similar_chunks = chunks_metadata.iloc[indices[0]]
        
        # Criar um conjunto para rastrear IDs originais já vistos
        seen_ids = set()
        unique_examples = []
        
        # Filtrar para obter comentários originais únicos
        for _, row in similar_chunks.iterrows():
            id_original = row["id_original"]
            if id_original not in seen_ids:
                seen_ids.add(id_original)
                unique_examples.append({
                    "ID_Original": id_original,
                    "Comentario_Original": row["comentario_original"],
                    "Similaridade": 1 - (distances[0][len(unique_examples)] / np.max(distances[0]) if np.max(distances[0]) > 0 else distances[0][len(unique_examples)])
                })
                
                # Limitar ao número de exemplos solicitado
                if len(unique_examples) >= min(k, len(similar_chunks)):
                    break
        
        return pd.DataFrame(unique_examples)
    
    except Exception as e:
        st.error(f"Erro na busca de exemplos similares: {e}")
        return pd.DataFrame()

# Função para analisar comentário com GPT-4 usando RAG
def analisar_comentario_openai_with_rag(pedido, comentario, index, chunks_metadata, num_exemplos=5, model="gpt-4"):
    try:
        # Gerar embedding para o comentário
        comentario_embedding = get_embedding(str(comentario))
        
        # Buscar exemplos similares
        similar_examples = retrieve_similar_examples(
            comentario_embedding, 
            index, 
            chunks_metadata, 
            k=num_exemplos
        )
        
        # Construir o prompt com os exemplos históricos
        exemplos_historicos = ""
        for i, (_, row) in enumerate(similar_examples.iterrows(), 1):
            exemplos_historicos += f"Exemplo {i} (OS: {row['ID_Original']}, Similaridade: {row['Similaridade']:.2f}):\n{row['Comentario_Original']}\n\n"
        
        system_message = """Você é um especialista em análise preditiva de qualidade para uma empresa de serviços automotivos (troca/reparo de vidros - VFLR, e funilaria/martelinho de ouro - RRSM). Sua função é prever a probabilidade de uma Ordem de Serviço (OS) gerar uma reclamação formal, com base em anotações de atendimento e exemplos históricos.

**Objetivo:** Classificar o risco de reclamação e fornecer uma análise detalhada.

**Fatores Preditivos Fundamentais (Peso de Influência na Probabilidade):**
1.  **Frequência de Contatos (Peso 4):** Indique no comentário se o cliente já realizou múltiplos contatos sobre a mesma OS.
    * 1 contato: baixo risco
    * 2 contatos: médio risco
    * 3+ contatos: risco elevado
2.  **Tempo de Espera (Peso 3):** Identifique atrasos ou esperas prolongadas.
    * Negociação Carglass: > 1 dia útil
    * Acompanhamento de peças (VFLR): > 5 dias úteis
    * Agendamento: > 1 dia útil
    * Confirmação de execução: qualquer atraso
3.  **Falhas Processuais (Peso 2):** Detecte erros que causem retrabalho ou frustração.
    * Cadastro incorreto (endereço/placa/modelo)
    * Solicitações específicas não atendidas
    * Falhas de comunicação entre setores
    * Problemas técnicos após execução do serviço (gravidade alta)
4.  **Estado Emocional do Cliente (Peso 1):** Procure por sinais de frustração, irritação, insatisfação ou exigências.

**Metodologia para Cálculo da Probabilidade e Classificação (Com base nos seus dados históricos):**
* Avalie a presença e a intensidade dos Fatores Preditivos Fundamentais inferidos do comentário e exemplos históricos.
* **Dê atenção especial às palavras-chave e padrões frequentemente encontrados em reclamações históricas:** `cliente`, `contato`, `sinistro`, `informa`, `veículo`, `aguardando`, `retorno`, `troca`, `serviço`, `data`, `guincho`, `assistência`, `execução`, `peça`, `segurado`, `local`, `confirma`, `horas`, `pedido`, `atendente`, e bigrams como `cliente informa`, `aguardando retorno`, `contato cliente`, `guincho assistência 24h`, `assistência 24h`. A presença destes termos, especialmente se combinados, aumenta significativamente o risco.
* Classifique o risco e a porcentagem:
    * Baixa: 0-30%
    * Média: 31-60%
    * Alta: 61-85%
    * Crítica: 86-100%

**Formato de Resposta Esperado (ESTRITAMENTE SEGUIR ESTE FORMATO):**
```
- Pedido: [NÚMERO_DA_OS_OU_N/A]
- Probabilidade de Reclamação: [Baixa/Média/Alta/Crítica]
- Porcentagem de Reclamação: [XX%]
- Fatores Críticos: [Liste os fatores (Frequência, Tempo, Falhas, Estado Emocional) que contribuíram para o risco, citando sinais específicos do comentário (incluindo as palavras-chave relevantes) e/ou dos exemplos históricos. Ex: "Atraso no tempo de espera (3 dias, palavra 'aguardando' presente), cliente demonstra irritação (inferido de tom e vocabulário), similar a caso histórico de 'atraso guincho'."]
- Conclusão: [Resumo conciso do risco e sugestão de ação preventiva se o risco for Médio ou superior. Ex: "Risco alto devido a múltiplos atrasos. Necessário contato imediato para oferecer solução e evitar escalada do sinistro."]
```"""

        user_message = f"""Analise o seguinte comentário de atendimento e determine a probabilidade de uma reclamação formal ser aberta:

PEDIDO/OS: {pedido}

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
            if "- Pedido:" in line:
                info["pedido"] = line.split(":", 1)[1].strip()
            elif "- Probabilidade de Reclamação:" in line:
                info["probabilidade"] = line.split(":", 1)[1].strip()
            elif "- Porcentagem de Reclamação:" in line:
                percentage_text = line.split(":", 1)[1].strip()
                percentage = ''.join(filter(lambda x: x.isdigit() or x == '.', percentage_text))
                info["porcentagem"] = float(percentage)
            elif "- Fatores Críticos:" in line:
                info["fatores"] = line.split(":", 1)[1].strip()
            elif "- Conclusão:" in line:
                info["conclusao"] = line.split(":", 1)[1].strip()
        
        return info
    except Exception as e:
        st.error(f"Erro ao extrair informações da resposta da IA: {e}")
        return {
            "pedido": "Erro",
            "probabilidade": "Erro",
            "porcentagem": 0,
            "fatores": "Erro na extração",
            "conclusao": "Não foi possível analisar a resposta da IA."
        }

# Função para formatar a exibição do resultado
def format_result_display(pedido, ai_response):
    info = extract_info_from_ai_response(ai_response)
    
    pedido_display = info.get("pedido", pedido)
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
        <h3>Pedido: {pedido_display}</h3>
        <p><strong>Probabilidade de Reclamação:</strong> <span class="{risk_class}">{probabilidade}</span></p>
        <p><strong>Porcentagem de Risco:</strong> <span class="{risk_class}">{porcentagem}%</span></p>
        <p><strong>Fatores Críticos:</strong> {fatores}</p>
        <p><strong>Conclusão:</strong> {conclusao}</p>
    </div>
    """
    return html

# Função principal
def main():
    # Carregar e indexar a base histórica
    with st.spinner("Carregando e indexando base histórica..."):
        index, chunks_metadata = load_and_index_historical_data_with_chunking()
        if index is None or chunks_metadata is None:
            st.error("❌ Erro ao carregar a base histórica. Verifique se o arquivo CSV está disponível.")
            st.stop()
    
    # Upload de arquivo
    st.subheader("📤 Upload de Arquivo")
    uploaded_file = st.file_uploader("Envie um arquivo Excel, PDF ou JSON com os atendimentos", type=["xlsx", "pdf", "json", "csv"])
    
    # Inicializa DataFrame fora do if para evitar NameError
    df = pd.DataFrame()
    
    if uploaded_file:
        with st.spinner("Processando arquivo..."):
            if uploaded_file.name.endswith((".xlsx", ".csv")):
                try:
                    if uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    else:  # CSV
                        df = pd.read_csv(uploaded_file)
                    
                    # Exibir informações sobre o arquivo
                    st.info(f"Arquivo carregado: {uploaded_file.name} | {df.shape[0]} linhas x {df.shape[1]} colunas")
                    
                    # Se for uma única linha com múltiplos campos, concatena
                    if df.shape[0] == 1 and df.shape[1] > 1:
                        comentario = ' '.join([str(v) for v in df.iloc[0].values if pd.notna(v)])
                        df = pd.DataFrame({"Pedido": ["Pedido 1"], "Comentario": [comentario]})
                    elif df.shape[1] == 1:
                        # Se for uma única coluna, assume que é o comentário
                        df.insert(0, "Pedido", [f"OS_Temp_{i+1}" for i in range(len(df))])
                        df.columns = ["Pedido", "Comentario"]
                    else:
                        # Permitir que o usuário selecione as colunas
                        colunas_disponiveis = df.columns.tolist()
                        
                        st.subheader("Seleção de Colunas")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            coluna_pedido = st.selectbox(
                                "Selecione a coluna de Pedido/OS:",
                                options=["Nenhuma"] + colunas_disponiveis,
                                index=0
                            )
                        
                        with col2:
                            coluna_comentario = st.selectbox(
                                "Selecione a coluna de Comentários:",
                                options=colunas_disponiveis,
                                index=0 if colunas_disponiveis else None
                            )
                        
                        if coluna_comentario:
                            if coluna_pedido != "Nenhuma":
                                df = df[[coluna_pedido, coluna_comentario]].rename(columns={coluna_pedido: "Pedido", coluna_comentario: "Comentario"})
                            else:
                                # Se não selecionou coluna de pedido, gera IDs temporários
                                df = df[[coluna_comentario]].rename(columns={coluna_comentario: "Comentario"})
                                df.insert(0, "Pedido", [f"OS_Temp_{i+1}" for i in range(len(df))])
                            
                            # Agrupa por pedido para evitar duplicatas
                            df["Comentario"] = df.groupby("Pedido")["Comentario"].transform(lambda x: '\n'.join(x.astype(str)))
                            df = df.drop_duplicates(subset=["Pedido"]).reset_index(drop=True)
                        else:
                            st.error("❌ Selecione a coluna de Comentário para continuar.")
                            st.stop()
                
                except Exception as e:
                    st.error(f"❌ Erro ao ler o arquivo: {e}")
                    st.stop()
            
            elif uploaded_file.name.endswith(".pdf"):
                df = extract_text_from_pdf(uploaded_file)
                # Adiciona uma coluna de "Pedido" fictícia para PDFs
                df.insert(0, "Pedido", [f"PDF_{i+1}" for i in range(len(df))])
            
            elif uploaded_file.name.endswith(".json"):
                df = extract_text_from_json(uploaded_file)
                # Adiciona uma coluna de "Pedido" fictícia para JSONs
                df.insert(0, "Pedido", [f"JSON_{i+1}" for i in range(len(df))])
        
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
                    
                    # Analisar com GPT-4 usando RAG
                    resultado = analisar_comentario_openai_with_rag(
                        pedido,
                        str(comentario),
                        index,
                        chunks_metadata,
                        num_exemplos=num_exemplos,
                        model=model
                    )
                    
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
