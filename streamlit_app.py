def analyze_with_gpt(self, text: str,import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import PyPDF2
import io
import os
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Nota: Arquivos mantidos para versões futuras:
# - Dados_SRO.pkl: Base histórica para implementação futura com embeddings/RAG
# - dados_semSRO.pkl: Base complementar para análises específicas
# Versão atual usa análise baseada em prompt estruturado

# Configuração da página
st.set_page_config(
    page_title="SRO Risk Analyzer - Prompt Version",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SROPromptAnalyzer:
    """Classe para análise de risco SRO usando prompt estruturado"""
    
    def __init__(self):
        self.client = None
        self.is_loaded = False
        
        # Palavras-chave para análise
        self.neutral_words = [
            'fila', 'data', 'equipe', 'atualização', 'agenda', 'recontato',
            'inserido', 'tabela', 'negociado', 'complemento', 'evento', 
            'telefone', 'inicial', 'hashtag', 'uf', 'id', 'solicitante',
            'complexidade', 'código', 'sku', 'whats', 'observação', 
            'pergunta', 'lojista', 'item', 'qt', 'escala', 'criação',
            'exclusão', 'tabelado', 'responsável', 'bloqueio', 'distribuidor',
            'anjos', 'isento', 'receptivo', 'tela', 'dedutível', 'incluído',
            'imports', 'estética', 'agradou', 'geral', 'objeto', 'vida'
        ]
        
        self.technical_issues = [
            'defeito', 'conserto', 'danos', 'sinistro', 'vazamento',
            'barulho', 'quebra', 'arranhado', 'sujo', 'manchado',
            'escorrida', 'descolado', 'solto', 'acendendo', 'parou',
            'sumiu', 'faltando', 'faltou', 'errado', 'errada',
            'incompleto', 'danificado', 'estragado', 'pior', 'voltou'
        ]
        
        self.negative_moderate = [
            'terrível', 'péssimo', 'horrível', 'decepcionado', 'frustrado',
            'reclamar', 'problema', 'erro', 'falha', 'demora', 'demorado',
            'insatisfeito', 'revoltado', 'indignado', 'absurdo', 'inaceitável'
        ]
        
        self.legal_risk = [
            'processar', 'advogado', 'jurídico', 'procon', 'denúncia',
            'órgão', 'fiscalização', 'consumidor', 'direito', 'prejuízo'
        ]
        
        self.positive_words = [
            'excelente', 'ótimo', 'perfeito', 'maravilhoso', 'fantástico',
            'agradecer', 'obrigado', 'parabéns', 'satisfeito', 'contente',
            'recomendo', 'eficiente', 'rápido', 'atencioso', 'prestativo'
        ]
        
    def load_system(self, api_key: str) -> bool:
        """Carrega o sistema OpenAI"""
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self.is_loaded = True
            return True
        except Exception as e:
            st.error(f"Erro ao carregar OpenAI: {str(e)}")
            return False
    
    def count_contacts(self, text: str) -> int:
        """Conta o número de contatos baseado em padrões textuais"""
        # Padrões que indicam múltiplos contatos
        contact_patterns = [
            r'contato\s*\d+',
            r'ligação\s*\d+',
            r'retorno\s*\d+',
            r'recontato',
            r'nova\s*tentativa',
            r'segundo\s*contato',
            r'terceiro\s*contato',
            r'quarto\s*contato'
        ]
        
        contacts = 1  # Pelo menos 1 contato
        text_lower = text.lower()
        
        for pattern in contact_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                contacts += len(matches)
        
        # Verificar palavras neutras para reduzir risco
        neutral_count = sum(1 for word in self.neutral_words if word in text_lower)
        if neutral_count > 3:  # Se muitas palavras neutras, reduzir contatos
            contacts = max(1, contacts - 1)
            
        return min(contacts, 5)  # Máximo 5 contatos
    
    def analyze_waiting_time(self, text: str) -> int:
        """Analisa tempo de espera baseado em padrões"""
        text_lower = text.lower()
        score = 0
        
        # Padrões de atraso
        delay_patterns = [
            'atras', 'demor', 'espera', 'aguard', 'pend',
            'mais de', 'já faz', 'há dias', 'semanas'
        ]
        
        urgent_patterns = [
            'urgente', 'emergência', 'pressa', 'rápido',
            'imediato', 'hoje', 'agora'
        ]
        
        for pattern in delay_patterns:
            if pattern in text_lower:
                score += 3
                
        for pattern in urgent_patterns:
            if pattern in text_lower:
                score += 2
                
        return min(score, 10)
    
    def analyze_operational_failures(self, text: str) -> int:
        """Analisa falhas operacionais"""
        text_lower = text.lower()
        score = 0
        
        # Indícios técnicos (alto risco)
        technical_count = sum(1 for word in self.technical_issues if word in text_lower)
        score += technical_count * 3
        
        # Falhas de processo (médio risco)
        process_patterns = [
            'cadastro incorreto', 'não atendidas', 'falha.*comunicação',
            'problema.*técnico', 'pós.*serviço'
        ]
        
        for pattern in process_patterns:
            if re.search(pattern, text_lower):
                score += 2
                
        return min(score, 10)
    
    def analyze_emotional_state(self, text: str) -> int:
        """Analisa estado emocional"""
        text_lower = text.lower()
        score = 0
        
        # Termos negativos moderados (1 ponto cada)
        negative_count = sum(1 for word in self.negative_moderate if word in text_lower)
        score += negative_count * 1
        
        # Termos de risco jurídico (3 pontos cada)
        legal_count = sum(1 for word in self.legal_risk if word in text_lower)
        score += legal_count * 3
        
        # Termos positivos reduzem risco (-1 ponto cada)
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        score -= positive_count * 1
        
        return max(0, min(score, 10))
    
    def calculate_risk_score(self, text: str, order_id: str = "N/A") -> Dict:
        """Calcula o score de risco baseado na metodologia do prompt"""
        
        # Analisar cada fator
        contacts = self.count_contacts(text)
        waiting_time = self.analyze_waiting_time(text)
        operational_failures = self.analyze_operational_failures(text)
        emotional_state = self.analyze_emotional_state(text)
        
        # Converter para scores 0-10
        contact_score = min(10, contacts * 2)  # 1=2, 2=4, 3+=6+
        waiting_score = waiting_time
        failure_score = operational_failures
        emotion_score = emotional_state
        
        # Aplicar pesos
        weighted_contacts = contact_score * 4      # Peso 4
        weighted_waiting = waiting_score * 3       # Peso 3
        weighted_failures = failure_score * 2      # Peso 2
        weighted_emotion = emotion_score * 1       # Peso 1
        
        # Calcular total (máximo 100)
        total_score = weighted_contacts + weighted_waiting + weighted_failures + weighted_emotion
        percentage = min(100, total_score)
        
        # Classificar risco
        if percentage >= 86:
            risk_level = "Crítico"
            risk_color = "#FF0000"
        elif percentage >= 61:
            risk_level = "Alto"
            risk_color = "#FF4B4B"
        elif percentage >= 31:
            risk_level = "Médio"
            risk_color = "#FF8C00"
        else:
            risk_level = "Baixo"
            risk_color = "#00C851"
        
        # Identificar fatores críticos
        critical_factors = []
        if contacts >= 3:
            critical_factors.append(f"{contacts} contatos")
        if any(word in text.lower() for word in self.legal_risk):
            critical_factors.append("ameaça jurídica")
        if any(word in text.lower() for word in self.technical_issues):
            critical_factors.append("problemas técnicos")
        if waiting_score > 5:
            critical_factors.append("atraso no atendimento")
        
        return {
            "order_id": order_id,
            "risk_level": risk_level,
            "percentage": percentage,
            "risk_color": risk_color,
            "factors": {
                "contacts": contacts,
                "contact_score": contact_score,
                "waiting_score": waiting_score,
                "failure_score": failure_score,
                "emotion_score": emotion_score
            },
            "weighted_scores": {
                "contacts": weighted_contacts,
                "waiting": weighted_waiting,
                "failures": weighted_failures,
                "emotion": weighted_emotion
            },
            "critical_factors": critical_factors,
            "total_score": total_score
        }
    
    def analyze_with_gpt(self, text: str, order_id: str = "N/A") -> Dict:
        """Análise usando GPT-4o com o prompt estruturado"""
        if not self.is_loaded:
            return {"error": "Sistema não carregado"}
        
        # Prompt estruturado baseado no PDF
        prompt = f"""
Você é um analista preditivo em uma empresa de serviços automotivos, especialista em prever o risco de um cliente formalizar uma reclamação (SRO - Sistema de Registro de Ocorrências).

Analise o seguinte registro de atendimento e estime a probabilidade (0% a 100%) de reclamação formal:

TEXTO PARA ANÁLISE:
{text}

METODOLOGIA DE ANÁLISE:
1. Frequência de Contatos (Peso 4): Conte indícios de múltiplos contatos
2. Tempo de Espera (Peso 3): Identifique atrasos ou urgência
3. Falhas Operacionais (Peso 2): Detecte problemas técnicos ou de processo
4. Estado Emocional (Peso 1): Avalie tom emocional e ameaças

PALAVRAS-CHAVE IMPORTANTES:
- Risco Jurídico (alto risco): processar, advogado, jurídico, procon, denúncia
- Problemas Técnicos: defeito, conserto, danos, vazamento, quebra, erro
- Sentimento Negativo: terrível, péssimo, frustrado, revoltado, absurdo
- Sentimento Positivo (reduz risco): excelente, ótimo, agradecer, satisfeito

IMPORTANTE: Responda APENAS com um número de 0 a 100 representando a porcentagem de risco, seguido de uma linha com a classificação (Baixo/Médio/Alto/Crítico), e depois uma explicação breve.

Exemplo de resposta:
75
Alto
Cliente demonstra múltiplos contatos e frustração evidente.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise preditiva de reclamações de clientes. Seja preciso e objetivo."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            # Extrair porcentagem e classificação do GPT
            lines = gpt_response.split('\n')
            try:
                gpt_percentage = float(lines[0])
                gpt_classification = lines[1] if len(lines) > 1 else "Indefinido"
                gpt_explanation = lines[2] if len(lines) > 2 else "Análise não disponível"
            except (ValueError, IndexError):
                # Fallback para análise local se GPT falhar
                local_analysis = self.calculate_risk_score(text, order_id)
                return local_analysis
            
            # Usar resultado do GPT como principal
            if gpt_percentage >= 86:
                risk_level = "Crítico"
                risk_color = "#FF0000"
            elif gpt_percentage >= 61:
                risk_level = "Alto"
                risk_color = "#FF4B4B"
            elif gpt_percentage >= 31:
                risk_level = "Médio"
                risk_color = "#FF8C00"
            else:
                risk_level = "Baixo"
                risk_color = "#00C851"
            
            # Fazer análise local para breakdown detalhado
            local_analysis = self.calculate_risk_score(text, order_id)
            
            # Identificar fatores críticos baseado na análise local
            critical_factors = []
            if local_analysis["factors"]["contacts"] >= 3:
                critical_factors.append(f"{local_analysis['factors']['contacts']} contatos")
            if any(word in text.lower() for word in self.legal_risk):
                critical_factors.append("ameaça jurídica")
            if any(word in text.lower() for word in self.technical_issues):
                critical_factors.append("problemas técnicos")
            if local_analysis["factors"]["waiting_score"] > 5:
                critical_factors.append("atraso no atendimento")
            
            return {
                "order_id": order_id,
                "risk_level": risk_level,
                "percentage": gpt_percentage,
                "risk_color": risk_color,
                "factors": local_analysis["factors"],  # Manter breakdown local
                "weighted_scores": local_analysis["weighted_scores"],
                "critical_factors": critical_factors,
                "total_score": gpt_percentage,
                "gpt_analysis": gpt_response,
                "gpt_percentage": gpt_percentage,
                "gpt_classification": gpt_classification,
                "gpt_explanation": gpt_explanation,
                "method": "gpt_primary"
            }
            
        except Exception as e:
            st.warning(f"Erro na análise GPT-4o: {str(e)}")
            # Fallback para análise local
            local_analysis = self.calculate_risk_score(text, order_id)
            return {
                **local_analysis,
                "gpt_analysis": f"Análise GPT-4o não disponível: {str(e)}",
                "method": "local_fallback"
            }

def extract_text_from_file(uploaded_file) -> str:
    """Extrai texto de arquivos uploaded"""
    try:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
            
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                          "application/vnd.ms-excel"]:
            df = pd.read_excel(uploaded_file, header=None)
            text_parts = []
            
            for index, row in df.iterrows():
                for cell_value in row:
                    if pd.notna(cell_value) and str(cell_value).strip():
                        cell_text = str(cell_value).strip()
                        if cell_text.startswith('"') and cell_text.endswith('"'):
                            cell_text = cell_text[1:-1]
                        text_parts.append(cell_text)
            
            combined_text = " | ".join(text_parts)
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."
            
            return combined_text
            
        elif file_type == "application/json":
            json_data = json.load(uploaded_file)
            text = json.dumps(json_data, ensure_ascii=False, indent=2)
            return text
            
        elif file_type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
            
        else:
            return "Tipo de arquivo não suportado"
            
    except Exception as e:
        return f"Erro ao extrair texto: {str(e)}"

def create_risk_gauge(risk_score: float, risk_level: str, risk_color: str):
    """Cria gauge de risco"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risco de Reclamação SRO"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': "#E8F5E8"},
                {'range': [30, 60], 'color': "#FFF8DC"},
                {'range': [60, 85], 'color': "#FFE4B5"},
                {'range': [85, 100], 'color': "#FFE4E1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_factors_chart(analysis_result: Dict):
    """Cria gráfico dos fatores de risco"""
    factors = analysis_result["factors"]
    
    factor_names = ["Contatos", "Tempo Espera", "Falhas Op.", "Estado Emoc."]
    scores = [
        factors["contact_score"],
        factors["waiting_score"], 
        factors["failure_score"],
        factors["emotion_score"]
    ]
    weights = [4, 3, 2, 1]
    weighted_scores = [s * w for s, w in zip(scores, weights)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Score Base',
        x=factor_names,
        y=scores,
        marker_color='lightblue',
        yaxis='y1'
    ))
    
    fig.add_trace(go.Bar(
        name='Score Ponderado',
        x=factor_names,
        y=weighted_scores,
        marker_color='darkblue',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Análise por Fatores",
        xaxis=dict(title="Fatores"),
        yaxis=dict(title="Score Base (0-10)", side="left", range=[0, 10]),
        yaxis2=dict(title="Score Ponderado", side="right", overlaying="y", range=[0, 40]),
        barmode='group',
        height=400
    )
    
    return fig

def analyze_text(analyzer: SROPromptAnalyzer, text: str, source_name: str, order_id: str = "N/A"):
    """Função para analisar texto e mostrar resultados"""
    
    with st.spinner("🤖 Analisando risco de reclamação SRO..."):
        result = analyzer.analyze_with_gpt(text, order_id)
    
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Resultado da Análise")
        
        # Gauge de risco
        gauge_fig = create_risk_gauge(
            result["percentage"], 
            result["risk_level"], 
            result["risk_color"]
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Métricas principais
        st.metric("📈 Score de Risco", f"{result['percentage']:.1f}%")
        st.metric("🏷️ Classificação", result["risk_level"])
        st.metric("📞 Número de Contatos", result["factors"]["contacts"])
        
    with col2:
        st.subheader("📈 Análise Detalhada por Fatores")
        
        # Gráfico de fatores
        factors_chart = create_factors_chart(result)
        st.plotly_chart(factors_chart, use_container_width=True)
    
    # Análise GPT
    if "gpt_analysis" in result:
        st.subheader("🤖 Análise Detalhada (GPT-4o)")
        
        # Mostrar resultado estruturado se disponível
        if result.get("method") == "gpt_primary":
            col_gpt1, col_gpt2 = st.columns(2)
            with col_gpt1:
                st.metric("🎯 GPT-4o Score", f"{result.get('gpt_percentage', 0):.1f}%")
            with col_gpt2:
                st.metric("🏷️ GPT-4o Classificação", result.get('gpt_classification', 'N/A'))
            
            if result.get('gpt_explanation'):
                st.info(f"**Explicação**: {result['gpt_explanation']}")
        
        # Análise completa
        st.text_area(
            "Análise completa:",
            result["gpt_analysis"],
            height=150,
            disabled=True
        )
    
    # Fatores críticos identificados
    if result["critical_factors"]:
        st.subheader("⚠️ Fatores Críticos Identificados")
        for factor in result["critical_factors"]:
            st.warning(f"• {factor}")
    
    # Breakdown detalhado
    with st.expander("🔍 Breakdown Detalhado da Análise"):
        st.write("**Scores por Fator:**")
        factors = result["factors"]
        weighted = result["weighted_scores"]
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.write("**Fator**")
            st.write("Contatos")
            st.write("Tempo Espera")
            st.write("Falhas Op.")
            st.write("Estado Emoc.")
            
        with col_b:
            st.write("**Score Base**")
            st.write(f"{factors['contact_score']}/10")
            st.write(f"{factors['waiting_score']}/10")
            st.write(f"{factors['failure_score']}/10")
            st.write(f"{factors['emotion_score']}/10")
            
        with col_c:
            st.write("**Score Ponderado**")
            st.write(f"{weighted['contacts']}/40")
            st.write(f"{weighted['waiting']}/30")
            st.write(f"{weighted['failures']}/20")
            st.write(f"{weighted['emotion']}/10")
        
        # Breakdown detalhado
        st.write(f"**Total GPT-4o: {result['percentage']:.1f}% = {result['risk_level']}**")
        
        # Comparação com análise local se disponível
        if result.get("method") == "gpt_primary":
            local_score = sum(result["weighted_scores"].values())
            st.write(f"*Comparação - Análise Local: {local_score:.1f}%*")
    
    # Recomendações baseadas no nível de risco
    st.subheader("💡 Recomendações de Ação")
    
    if result["percentage"] >= 86:
        st.error("🚨 **RISCO CRÍTICO**: Ação imediata necessária!")
        st.write("• Contato imediato com supervisor")
        st.write("• Priorização máxima do caso")
        st.write("• Antecipação de agendamento")
    elif result["percentage"] >= 61:
        st.warning("⚠️ **RISCO ALTO**: Monitoramento próximo!")
        st.write("• Contato proativo com o cliente")
        st.write("• Envolvimento do gestor técnico")
        st.write("• Feedback técnico imediato")
    elif result["percentage"] >= 31:
        st.info("ℹ️ **RISCO MÉDIO**: Atenção preventiva!")
        st.write("• Acompanhamento regular")
        st.write("• Correção de falhas identificadas")
        st.write("• Comunicação proativa")
    else:
        st.success("✅ **RISCO BAIXO**: Situação controlada!")
        st.write("• Acompanhamento padrão")
        st.write("• Manutenção da qualidade")
    
    # Download do relatório
    st.subheader("📥 Download do Relatório")
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "pedido": result["order_id"],
        "texto_analisado": text[:500] + "..." if len(text) > 500 else text,
        "resultado": {
            "nivel_risco": result["risk_level"],
            "porcentagem": result["percentage"],
            "fatores_criticos": result["critical_factors"]
        },
        "analise_detalhada": result.get("gpt_analysis", ""),
        "breakdown": {
            "fatores": result["factors"],
            "scores_ponderados": result["weighted_scores"],
            "total": result["total_score"]
        }
    }
    
    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📄 Baixar Relatório JSON",
        data=report_json,
        file_name=f"relatorio_sro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Interface Streamlit
def main():
    # Header
    st.title("🔍 SRO Risk Analyzer - Versão Prompt")
    st.markdown("**Sistema de Análise Preditiva de Reclamações**")
    st.markdown("*Baseado na metodologia de análise estruturada com 4 fatores ponderados*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Verificar API Key nos secrets
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.success("🔑 API Key carregada dos secrets")
        except KeyError:
            st.error("🔑 API Key não encontrada nos secrets")
            st.info("""
            **Configure nos Secrets do Streamlit:**
            1. Vá em Settings > Secrets
            2. Adicione: OPENAI_API_KEY = "sua_chave_aqui"
            """)
            api_key = None
        
        # Informações da metodologia
        st.header("📋 Metodologia de Análise")
        st.info("""
        **Fatores Ponderados:**
        
        🔢 **Frequência Contatos** (Peso 4)
        - 1 contato: baixo risco
        - 2 contatos: médio risco  
        - 3+ contatos: alto risco
        
        ⏰ **Tempo de Espera** (Peso 3)
        - Atrasos e urgência
        
        ⚙️ **Falhas Operacionais** (Peso 2)
        - Problemas técnicos
        - Falhas de processo
        
        😠 **Estado Emocional** (Peso 1)
        - Sentimento negativo
        - Ameaças jurídicas
        - Palavras positivas (reduzem risco)
        """)
        
        st.header("🎯 Classificação de Risco")
        st.write("• **Baixo**: 0-30%")
        st.write("• **Médio**: 31-60%") 
        st.write("• **Alto**: 61-85%")
        st.write("• **Crítico**: 86-100%")
    
    # Verificar pré-requisitos
    if not api_key:
        st.error("🔑 API Key não configurada nos secrets do Streamlit")
        st.stop()
    
    # Inicializar analyzer
    @st.cache_resource
    def load_analyzer(_api_key):
        analyzer = SROPromptAnalyzer()
        if analyzer.load_system(_api_key):
            return analyzer
        return None
    
    with st.spinner("🤖 Carregando sistema de análise..."):
        analyzer = load_analyzer(api_key)
    
    if analyzer is None:
        st.error("❌ Falha ao carregar o sistema SRO")
        st.stop()
    
    st.success("✅ Sistema SRO carregado com sucesso!")
    
    # Interface principal
    tab1, tab2, tab3 = st.tabs(["📤 Upload de Arquivo", "✍️ Texto Manual", "🧪 Exemplos de Teste"])
    
    with tab1:
        st.header("📤 Análise de Arquivo")
        st.markdown("Faça upload de um arquivo para analisar o risco de reclamação")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo",
            type=['pdf', 'xlsx', 'xls', 'json', 'txt'],
            help="Formatos suportados: PDF, Excel, JSON, TXT"
        )
        
        order_id = st.text_input("ID do Pedido (opcional)", placeholder="ORD123456")
        
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
                analyze_text(analyzer, extracted_text, uploaded_file.name, order_id or "Arquivo")
    
    with tab2:
        st.header("✍️ Análise de Texto Manual")
        st.markdown("Digite ou cole um texto para analisar")
        
        order_id = st.text_input("ID do Pedido", placeholder="ORD123456", key="manual_order")
        
        manual_text = st.text_area(
            "Digite o texto para análise:",
            height=200,
            placeholder="Cole aqui o registro de atendimento que deseja analisar..."
        )
        
        if manual_text and st.button("🔍 Analisar Risco", key="analyze_manual"):
            analyze_text(analyzer, manual_text, "Texto Manual", order_id or "Manual")
    
    with tab3:
        st.header("🧪 Exemplos de Teste")
        st.markdown("Teste o sistema com exemplos pré-definidos")
        
        examples = {
            "Baixo Risco": "Cliente agradeceu pelo excelente atendimento. Serviço executado conforme combinado. Cliente satisfeito com o resultado.",
            
            "Médio Risco": "Cliente ligou duas vezes perguntando sobre o andamento. Mencionou que está com pressa para viajar. Aguardando retorno há 2 dias.",
            
            "Alto Risco": "Terceiro contato do cliente. Reclamou que o defeito voltou após o conserto. Disse que está muito frustrado e decepcionado com o serviço.",
            
            "Crítico": "Cliente extremamente revoltado. Quarto contato! Disse que vai acionar o Procon e processar a empresa. Defeito persiste e está causando prejuízo. Inaceitável!"
        }
        
        selected_example = st.selectbox("Escolha um exemplo:", list(examples.keys()))
        
        if st.button("🔍 Testar Exemplo", key="test_example"):
            st.write(f"**Testando: {selected_example}**")
            st.write(f"*Texto:* {examples[selected_example]}")
            st.markdown("---")
            analyze_text(analyzer, examples[selected_example], "Exemplo", f"TESTE_{selected_example.replace(' ', '_').upper()}")
        
        # Exemplo customizado
        st.subheader("📝 Criar Exemplo Customizado")
        
        custom_order = st.text_input("ID do Pedido Teste", placeholder="TESTE_001", key="custom_order")
        custom_text = st.text_area(
            "Texto de exemplo:",
            height=150,
            placeholder="Digite aqui um exemplo personalizado para testar...",
            key="custom_example"
        )
        
        if custom_text and st.button("🔍 Testar Customizado", key="test_custom"):
            analyze_text(analyzer, custom_text, "Exemplo Customizado", custom_order or "TESTE_CUSTOM")

if __name__ == "__main__":
    main()
