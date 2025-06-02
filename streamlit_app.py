import streamlit as st
import pandas as pd
import numpy as np
import openai
import faiss
import pickle
import json
import PyPDF2
import io
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Configuração da página
st.set_page_config(
    page_title="SRO Risk Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SROAnalyzer:
    """Classe principal para análise de risco de reclamações SRO"""
    
    def __init__(self):
        self.faiss_index = None
        self.data_list = None
        self.client = None
        self.is_loaded = False
        
    def load_system(self, api_key: str) -> bool:
        """Carrega o sistema FAISS e OpenAI"""
        try:
            # Configurar OpenAI
            self.client = openai.OpenAI(api_key=api_key)
            
            # Carregar índice FAISS
            self.faiss_index = faiss.read_index("Embeddings_SRO.index")
            
            # Carregar dados correspondentes
            with open("Dados_SRO.pkl", 'rb') as f:
                self.data_list = pickle.load(f)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar sistema: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Gera embedding para um texto"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Erro ao gerar embedding: {str(e)}")
            return None
    
    def analyze_risk(self, text: str, top_k: int = 10) -> Dict:
        """Analisa risco de reclamação baseado em similaridade"""
        if not self.is_loaded:
            return {"error": "Sistema não carregado"}
        
        # Gerar embedding do texto
        embedding = self.generate_embedding(text)
        if embedding is None:
            return {"error": "Falha ao gerar embedding"}
        
        # Buscar similares
        query_vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        # Analisar resultados
        similarities = distances[0]
        max_similarity = float(similarities[0]) if len(similarities) > 0 else 0.0
        avg_similarity = float(np.mean(similarities))
        
        # Calcular score de risco (0-100%)
        risk_score = min(100, max_similarity * 100)
        
        # Classificar risco
        if risk_score >= 80:
            risk_level = "Alta"
            risk_color = "#FF4B4B"
        elif risk_score >= 60:
            risk_level = "Média"
            risk_color = "#FF8C00"
        elif risk_score >= 30:
            risk_level = "Baixa"
            risk_color = "#FFD700"
        else:
            risk_level = "Nula"
            risk_color = "#00C851"
        
        # Obter reclamações similares
        similar_complaints = []
        for i, (sim, idx) in enumerate(zip(similarities, indices[0])):
            if idx < len(self.data_list):
                item = self.data_list[idx].copy()
                item['similaridade'] = float(sim)
                item['rank'] = i + 1
                similar_complaints.append(item)
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "similar_complaints": similar_complaints,
            "total_analyzed": len(similarities)
        }

def extract_text_from_file(uploaded_file) -> str:
    """Extrai texto de arquivos uploaded"""
    try:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            # Extrair texto de PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
            
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                          "application/vnd.ms-excel"]:
            # Extrair texto de Excel
            df = pd.read_excel(uploaded_file)
            text = ""
            for col in df.columns:
                text += f"{col}: {' '.join(df[col].astype(str).tolist())}\n"
            return text
            
        elif file_type == "application/json":
            # Extrair texto de JSON
            json_data = json.load(uploaded_file)
            text = json.dumps(json_data, ensure_ascii=False, indent=2)
            return text
            
        elif file_type == "text/plain":
            # Arquivo de texto
            return str(uploaded_file.read(), "utf-8")
            
        else:
            return "Tipo de arquivo não suportado"
            
    except Exception as e:
        return f"Erro ao extrair texto: {str(e)}"

def create_risk_gauge(risk_score: float, risk_level: str, risk_color: str):
    """Cria gauge de risco"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risco de Reclamação"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': "#E8F5E8"},
                {'range': [30, 60], 'color': "#FFF8DC"},
                {'range': [60, 80], 'color': "#FFE4B5"},
                {'range': [80, 100], 'color': "#FFE4E1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_similarity_chart(similar_complaints: List[Dict]):
    """Cria gráfico de similaridade"""
    if not similar_complaints:
        return None
    
    df = pd.DataFrame(similar_complaints[:5])  # Top 5
    
    fig = px.bar(
        df, 
        x='rank', 
        y='similaridade',
        title="Top 5 Reclamações Similares",
        labels={'similaridade': 'Similaridade (%)', 'rank': 'Ranking'},
        color='similaridade',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400)
    return fig

def download_report(analysis_result: Dict, original_text: str) -> str:
    """Gera relatório para download"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "texto_analisado": original_text[:500] + "..." if len(original_text) > 500 else original_text,
        "analise_risco": {
            "score_risco": analysis_result["risk_score"],
            "nivel_risco": analysis_result["risk_level"],
            "similaridade_maxima": analysis_result["max_similarity"],
            "similaridade_media": analysis_result["avg_similarity"]
        },
        "reclamacoes_similares": analysis_result["similar_complaints"][:5]
    }
    
    return json.dumps(report, ensure_ascii=False, indent=2)

# Interface Streamlit
def main():
    # Header
    st.title("🔍 SRO Risk Analyzer")
    st.markdown("**Sistema de Análise de Risco de Reclamações**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configuração da API Key
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Insira sua chave da API OpenAI"
        )
        
        # Verificação dos arquivos
        st.header("📁 Status do Sistema")
        
        try:
            # Verificar se arquivos existem
            import os
            files_status = {
                "Embeddings_SRO.index": os.path.exists("Embeddings_SRO.index"),
                "Dados_SRO.pkl": os.path.exists("Dados_SRO.pkl")
            }
            
            for file, exists in files_status.items():
                icon = "✅" if exists else "❌"
                st.write(f"{icon} {file}")
            
            all_files_exist = all(files_status.values())
            
        except:
            all_files_exist = False
            st.error("❌ Arquivos SRO não encontrados")
        
        # Status geral
        if all_files_exist and api_key:
            st.success("🟢 Sistema Pronto")
        else:
            st.warning("🟡 Configuração Incompleta")
    
    # Verificar se pode prosseguir
    if not api_key:
        st.error("🔑 Por favor, configure sua OpenAI API Key na barra lateral")
        st.stop()
    
    if not all_files_exist:
        st.error("📁 Arquivos SRO não encontrados. Certifique-se de que os arquivos estão no diretório correto.")
        st.stop()
    
    # Inicializar analyzer
    @st.cache_resource
    def load_analyzer():
        analyzer = SROAnalyzer()
        if analyzer.load_system(api_key):
            return analyzer
        return None
    
    analyzer = load_analyzer()
    
    if analyzer is None:
        st.error("❌ Falha ao carregar o sistema SRO")
        st.stop()
    
    st.success("✅ Sistema SRO carregado com sucesso!")
    
    # Interface principal
    tab1, tab2 = st.tabs(["📤 Upload de Arquivo", "✍️ Texto Manual"])
    
    with tab1:
        st.header("📤 Análise de Arquivo")
        st.markdown("Faça upload de um arquivo para analisar o risco de reclamação")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo",
            type=['pdf', 'xlsx', 'xls', 'json', 'txt'],
            help="Formatos suportados: PDF, Excel, JSON, TXT"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Extraindo texto do arquivo..."):
                extracted_text = extract_text_from_file(uploaded_file)
            
            # Mostrar preview do texto
            with st.expander("👁️ Preview do Texto Extraído"):
                st.text_area(
                    "Texto extraído:",
                    extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                    height=200
                )
            
            if st.button("🔍 Analisar Risco", key="analyze_file"):
                analyze_text(analyzer, extracted_text, uploaded_file.name)
    
    with tab2:
        st.header("✍️ Análise de Texto Manual")
        st.markdown("Digite ou cole um texto para analisar")
        
        manual_text = st.text_area(
            "Digite o texto para análise:",
            height=200,
            placeholder="Cole aqui o texto que deseja analisar..."
        )
        
        if manual_text and st.button("🔍 Analisar Risco", key="analyze_manual"):
            analyze_text(analyzer, manual_text, "Texto Manual")

def analyze_text(analyzer: SROAnalyzer, text: str, source_name: str):
    """Função para analisar texto e mostrar resultados"""
    
    with st.spinner("🤖 Analisando risco de reclamação..."):
        result = analyzer.analyze_risk(text, top_k=10)
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Resultado da Análise")
        
        # Gauge de risco
        gauge_fig = create_risk_gauge(
            result["risk_score"], 
            result["risk_level"], 
            result["risk_color"]
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Métricas
        st.metric("📈 Score de Risco", f"{result['risk_score']:.1f}%")
        st.metric("🏷️ Classificação", result["risk_level"])
        st.metric("🔗 Similaridade Máxima", f"{result['max_similarity']:.3f}")
    
    with col2:
        st.subheader("📈 Análise de Similaridade")
        
        # Gráfico de similaridade
        if result["similar_complaints"]:
            sim_chart = create_similarity_chart(result["similar_complaints"])
            if sim_chart:
                st.plotly_chart(sim_chart, use_container_width=True)
        
        # Estatísticas
        st.write("**📊 Estatísticas:**")
        st.write(f"• Reclamações analisadas: {result['total_analyzed']}")
        st.write(f"• Similaridade média: {result['avg_similarity']:.3f}")
    
    # Detalhes das reclamações similares
    st.subheader("🔍 Reclamações Similares Encontradas")
    
    if result["similar_complaints"]:
        for i, complaint in enumerate(result["similar_complaints"][:5]):
            with st.expander(f"#{i+1} - Similaridade: {complaint['similaridade']:.1%}"):
                st.write("**Reclamação:**")
                st.write(complaint['reclamacao'])
                st.write("**Solução:**")
                st.write(complaint['solucao'])
    else:
        st.info("Nenhuma reclamação similar encontrada")
    
    # Download do relatório
    st.subheader("📥 Download do Relatório")
    
    report_json = download_report(result, text)
    
    st.download_button(
        label="📄 Baixar Relatório JSON",
        data=report_json,
        file_name=f"relatorio_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    # Recomendações
    st.subheader("💡 Recomendações")
    
    if result["risk_score"] >= 80:
        st.error("🚨 **RISCO ALTO**: Atenção imediata necessária. Implementar medidas preventivas urgentes.")
    elif result["risk_score"] >= 60:
        st.warning("⚠️ **RISCO MÉDIO**: Monitoramento contínuo recomendado. Considerar ações preventivas.")
    elif result["risk_score"] >= 30:
        st.info("ℹ️ **RISCO BAIXO**: Monitoramento regular suficiente.")
    else:
        st.success("✅ **RISCO NULO**: Situação controlada. Manter procedimentos padrão.")

if __name__ == "__main__":
    main()
