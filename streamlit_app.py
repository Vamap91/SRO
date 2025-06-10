import streamlit as st
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

# Configuração da página
st.set_page_config(
    page_title="SRO Risk Analyzer - Teste do Prompt",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SROPromptAnalyzer:
    """Classe para testar o prompt EXATO do PDF anexado"""
    
    def __init__(self):
        self.client = None
        self.is_loaded = False
        
    def load_system(self, api_key: str) -> bool:
        """Carrega o sistema OpenAI"""
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self.is_loaded = True
            return True
        except Exception as e:
            st.error(f"Erro ao carregar OpenAI: {str(e)}")
            return False
    
    def analyze_with_exact_prompt(self, text: str, order_id: str = "N/A") -> Dict:
        """Análise usando EXATAMENTE o prompt do PDF anexado"""
        if not self.is_loaded:
            return {"error": "Sistema não carregado"}
        
        try:
            # PROMPT EXATO BASEADO NO PDF ANEXADO
            prompt = f"""Role and Objective (in English)
You are a predictive quality analyst in an automotive service company.
Your task is to analyze service records written in Brazilian Portuguese and 
estimate the probability (0% to 100%) that a customer will file a formal complaint 
(SRO – Sistema de Registro de Ocorrências).
You must identify early warning signs, detect emotional tone, and score risk 
based on four weighted factors. Then, generate a final risk classification and a 
suggested preventive action, strictly following the expected output format (in 
Portuguese).

Contexto da Missão (em português)
Você é um analista preditivo em uma empresa de serviços automotivos, 
especialista em prever o risco de um cliente formalizar uma reclamação (SRO -
Sistema de Registro de Ocorrências) com base em ordens de serviço (OS) ainda 
em andamento.

Seu papel é identificar sinais precoces de insatisfação a partir do histórico 
textual de atendimentos, quantificar o risco de forma objetiva (0 a 100%) e 
recomendar ações preventivas.

Fatores Preditivos e Pesos
1. Frequência de Contatos – Peso 4
- 1 contato: risco baixo
- 2 contatos: risco médio
- 3 ou mais contatos: risco elevado

Atenuação contextual: se os múltiplos contatos contêm palavras neutras de 
acompanhamento, o risco é reduzido.

Palavras neutras (SEM-SRO): fila, data, equipe, atualização, agenda, recontato, 
inserido, tabela, negociado, complemento, evento, telefone, inicial, hashtag, uf, 
id, solicitante, complexidade, código, sku, whats, observação, pergunta, lojista, 
item, qt, escala, criação, exclusão, tabelado, responsável, bloqueio, distribuidor, 
anjos, isento, receptivo, tela, dedutível, incluído, imports, estética, agradou, 
geral, objeto, vida

2. Tempo de Espera – Peso 3
- Negociação Carglass: até 1 dia útil
- Peças (VFLR): até 5 dias úteis
- Agendamento: até 1 dia útil
- Execução: sem atrasos tolerados

3. Falhas Operacionais – Peso 2
A. Indícios técnicos (alto risco): defeito, conserto, danos, sinistro, vazamento, 
barulho, quebra, arranhado, sujo, manchado, escorrida, descolado, solto, 
acendendo, parou, sumiu, faltando, faltou, errado, errada, incompleto, 
danificado, estragado, pior, voltou

B. Falhas de processo (médio risco): cadastro incorreto, solicitações não 
atendidas, falhas de comunicação, problemas técnicos pós-serviço

4. Estado Emocional – Peso 1
Termos negativos moderados (1 ponto): terrível, péssimo, horrível, 
decepcionado, frustrado, reclamar, problema, erro, falha, demora, demorado, 
insatisfeito, revoltado, indignado, absurdo, inaceitável

Termos de risco jurídico (3 pontos): processar, advogado, jurídico, procon, 
denúncia, órgão, fiscalização, consumidor, direito, prejuízo

Termos positivos que reduzem risco (-1 ponto): excelente, ótimo, perfeito, 
maravilhoso, fantástico, agradecer, obrigado, parabéns, satisfeito, contente, 
recomendo, eficiente, rápido, atencioso, prestativo

Metodologia de Cálculo
1. Atribua um score (0 a 10) para cada fator, com base nas regras acima.
2. Multiplique cada score pelo peso do fator.
3. Some os valores ponderados para obter um total (máximo = 100 pontos).
4. Converta em percentual e classifique:
- Baixo: 0–30%
- Médio: 31–60%
- Alto: 61–85%
- Crítico: 86–100%

Formato Esperado de Saída (em português)
- Pedido: {order_id}
- Probabilidade de Reclamação: [Baixo/Médio/Alto/Crítico]
- Porcentagem Estimada: [X%]
- Fatores Críticos: [liste os principais fatores de risco identificados]
- Conclusão: [análise detalhada e recomendação de ação]

Sugestões de Ação (para risco ≥ Médio)
- Acione o cliente proativamente
- Priorize o caso com gestor técnico ou supervisor
- Antecipe agendamento e envie feedback técnico imediato
- Corrija falhas de cadastro ou comunicação antes do retorno do cliente
- Reforce canais de resolução rápida para evitar judicialização

TEXTO PARA ANÁLISE:
{text}

Analise o texto acima seguindo EXATAMENTE a metodologia descrita e forneça a resposta no formato especificado."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise preditiva de reclamações de clientes seguindo metodologia específica."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            # Inicializar variáveis com valores padrão
            pedido = order_id if order_id != "N/A" else "Não informado"
            probabilidade = "Indefinido"
            porcentagem = 0.0
            fatores_criticos = []
            conclusao = "Análise não disponível"
            
            # Processar resposta linha por linha de forma mais robusta
            if gpt_response:
                lines = gpt_response.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    if "pedido:" in line.lower():
                        pedido = line.split(":", 1)[1].strip()
                    elif "probabilidade de reclamação:" in line.lower():
                        probabilidade = line.split(":", 1)[1].strip()
                    elif "porcentagem estimada:" in line.lower():
                        porcentagem_text = line.split(":", 1)[1].strip().replace("%", "").replace(",", ".")
                        try:
                            # Extrair apenas números da string
                            import re
                            numbers = re.findall(r'\d+\.?\d*', porcentagem_text)
                            if numbers:
                                porcentagem = float(numbers[0])
                        except (ValueError, IndexError):
                            porcentagem = 0.0
                    elif "fatores críticos:" in line.lower():
                        fatores_text = line.split(":", 1)[1].strip()
                        if fatores_text and fatores_text.lower() != "nenhum":
                            fatores_criticos = [f.strip() for f in fatores_text.split(",") if f.strip()]
                    elif "conclusão:" in line.lower():
                        conclusao = line.split(":", 1)[1].strip()
            
            # Determinar cor e nível baseado na porcentagem
            if porcentagem >= 86:
                risk_color = "#FF0000"
                risk_level = "Crítico"
            elif porcentagem >= 61:
                risk_color = "#FF4B4B"
                risk_level = "Alto"
            elif porcentagem >= 31:
                risk_color = "#FF8C00"
                risk_level = "Médio"
            else:
                risk_color = "#00C851"
                risk_level = "Baixo"
            
            return {
                "order_id": pedido,
                "risk_level": risk_level,
                "percentage": porcentagem,
                "risk_color": risk_color,
                "critical_factors": fatores_criticos,
                "conclusion": conclusao,
                "gpt_analysis": gpt_response,
                "method": "prompt_exato",
                "prompt_version": "PDF_Anexado",
                "success": True
            }
            
        except Exception as e:
            st.error(f"Erro na análise GPT-4o: {str(e)}")
            return {
                "error": f"Falha na análise: {str(e)}",
                "order_id": order_id,
                "risk_level": "Erro",
                "percentage": 0.0,
                "risk_color": "#808080",
                "critical_factors": [],
                "conclusion": f"Erro durante a análise: {str(e)}",
                "gpt_analysis": "Análise não executada devido a erro",
                "method": "erro",
                "success": False
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
            if len(combined_text) > 5000:
                combined_text = combined_text[:5000] + "..."
            
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

def analyze_text_with_exact_prompt(analyzer: SROPromptAnalyzer, text: str, source_name: str, order_id: str = "N/A"):
    """Função para analisar texto usando o prompt EXATO do PDF"""
    
    with st.spinner("🤖 Testando prompt EXATO do PDF anexado..."):
        result = analyzer.analyze_with_exact_prompt(text, order_id)
    
    # Verificar se houve erro
    if not result.get("success", False):
        st.error(f"❌ Erro na análise: {result.get('error', 'Erro desconhecido')}")
        return
    
    # Header com informações do teste
    st.success("✅ **TESTE DO PROMPT EXATO DO PDF EXECUTADO**")
    st.info(f"**Versão do Prompt:** {result.get('prompt_version', 'N/A')} | **Método:** {result.get('method', 'N/A')}")
    
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
        st.metric("📋 Pedido", result["order_id"])
        
    with col2:
        st.subheader("⚠️ Fatores Críticos Identificados")
        
        if result["critical_factors"]:
            for i, factor in enumerate(result["critical_factors"], 1):
                st.warning(f"**{i}.** {factor}")
        else:
            st.info("Nenhum fator crítico identificado")
        
        st.subheader("💡 Conclusão")
        if result["conclusion"]:
            st.write(result["conclusion"])
        else:
            st.info("Conclusão não disponível")
    
    # Resposta completa do GPT-4o
    st.subheader("🤖 Resposta Completa do GPT-4o (Prompt Exato)")
    st.code(result["gpt_analysis"], language="text")
    
    # Verificação da metodologia
    st.subheader("🔍 Verificação da Metodologia")
    
    col_check1, col_check2 = st.columns(2)
    
    with col_check1:
        st.write("**✅ Elementos Obrigatórios Presentes:**")
        checks = [
            ("Pedido identificado", bool(result.get("order_id") and result["order_id"] != "Não informado")),
            ("Probabilidade classificada", bool(result.get("risk_level") and result["risk_level"] != "Indefinido")),
            ("Porcentagem calculada", result.get("percentage", 0) > 0),
            ("Fatores críticos listados", bool(result.get("critical_factors"))),
            ("Conclusão fornecida", bool(result.get("conclusion") and result["conclusion"] != "Análise não disponível"))
        ]
        
        for check_name, check_result in checks:
            if check_result:
                st.success(f"✅ {check_name}")
            else:
                st.error(f"❌ {check_name}")
    
    with col_check2:
        st.write("**📋 Classificação de Risco:**")
        st.write("• **Baixo**: 0-30%")
        st.write("• **Médio**: 31-60%") 
        st.write("• **Alto**: 61-85%")
        st.write("• **Crítico**: 86-100%")
        
        # Verificar se a classificação está correta
        expected_classification = ""
        if result["percentage"] >= 86:
            expected_classification = "Crítico"
        elif result["percentage"] >= 61:
            expected_classification = "Alto"
        elif result["percentage"] >= 31:
            expected_classification = "Médio"
        else:
            expected_classification = "Baixo"
        
        if result["risk_level"] == expected_classification:
            st.success(f"✅ Classificação correta: {result['risk_level']}")
        else:
            st.error(f"❌ Classificação incorreta: {result['risk_level']} (esperado: {expected_classification})")
    
    # Download do relatório de teste
    st.subheader("📥 Download do Relatório de Teste")
    
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "teste_prompt": "PDF_Anexado_Exato",
        "texto_analisado": text[:500] + "..." if len(text) > 500 else text,
        "resultado_prompt": {
            "pedido": result["order_id"],
            "probabilidade": result["risk_level"],
            "porcentagem": result["percentage"],
            "fatores_criticos": result["critical_factors"],
            "conclusao": result["conclusion"]
        },
        "resposta_completa_gpt": result["gpt_analysis"],
        "verificacao_metodologia": {
            "classificacao_esperada": expected_classification,
            "classificacao_obtida": result["risk_level"],
            "classificacao_correta": result["risk_level"] == expected_classification
        }
    }
    
    report_json = json.dumps(test_report, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📄 Baixar Relatório de Teste JSON",
        data=report_json,
        file_name=f"teste_prompt_sro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Interface Streamlit
def main():
    # Header
    st.title("🧪 SRO Risk Analyzer - TESTE DO PROMPT EXATO")
    st.markdown("**Testando a implementação EXATA do prompt do PDF anexado**")
    st.markdown("*Esta versão usa o prompt COMPLETO e LITERAL do documento fornecido*")
    
    # Warning sobre o objetivo
    st.warning("""
    ⚠️ **ATENÇÃO: Esta é uma versão de TESTE**
    
    O objetivo é verificar se o prompt do PDF anexado está funcional e produz resultados consistentes.
    Esta implementação usa o texto EXATO do PDF como prompt para o GPT-4o.
    """)
    
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
        
        # Informações do teste
        st.header("🧪 Informações do Teste")
        st.info("""
        **Prompt Testado:**
        - ✅ Texto COMPLETO do PDF
        - ✅ Metodologia dos 4 fatores
        - ✅ Pesos exatos (4,3,2,1)
        - ✅ Palavras-chave específicas
        - ✅ Formato de saída estruturado
        """)
        
        st.header("🎯 Classificação Esperada")
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
    
    with st.spinner("🤖 Carregando sistema de teste..."):
        analyzer = load_analyzer(api_key)
    
    if analyzer is None:
        st.error("❌ Falha ao carregar o sistema de teste")
        st.stop()
    
    st.success("✅ Sistema de teste carregado - Pronto para testar o prompt do PDF!")
    
    # Interface principal
    tab1, tab2, tab3 = st.tabs(["📤 Upload de Arquivo", "✍️ Texto Manual", "🧪 Exemplos de Teste"])
    
    with tab1:
        st.header("📤 Teste com Arquivo")
        st.markdown("Faça upload de um arquivo para testar o prompt")
        
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
            
            if st.button("🧪 TESTAR PROMPT EXATO", key="test_file"):
                analyze_text_with_exact_prompt(analyzer, extracted_text, uploaded_file.name, order_id or "ARQUIVO_TESTE")
    
    with tab2:
        st.header("✍️ Teste com Texto Manual")
        st.markdown("Digite ou cole um texto para testar o prompt")
        
        order_id = st.text_input("ID do Pedido", placeholder="ORD123456", key="manual_order")
        
        manual_text = st.text_area(
            "Digite o texto para teste:",
            height=200,
            placeholder="Cole aqui o registro de atendimento que deseja testar..."
        )
        
        if manual_text and st.button("🧪 TESTAR PROMPT EXATO", key="test_manual"):
            analyze_text_with_exact_prompt(analyzer, manual_text, "Texto Manual", order_id or "MANUAL_TESTE")
    
    with tab3:
        st.header("🧪 Exemplos Específicos para Teste")
        st.markdown("Teste o prompt com exemplos que devem produzir resultados específicos")
        
        examples = {
            "Teste Baixo Risco": {
                "text": "Cliente agradeceu pelo excelente atendimento. Serviço executado conforme combinado. Cliente muito satisfeito com o resultado e disse que recomenda a empresa.",
                "expected": "Baixo (0-30%)"
            },
            
            "Teste Médio Risco": {
                "text": "Cliente ligou duas vezes perguntando sobre o andamento do serviço. Mencionou que está com um pouco de pressa para viajar. Aguardando retorno há 2 dias. Demonstrou certa frustração.",
                "expected": "Médio (31-60%)"
            },
            
            "Teste Alto Risco": {
                "text": "Terceiro contato do cliente hoje. Reclamou que o defeito voltou após o conserto. Disse que está muito frustrado e decepcionado com o serviço. Mencionou problemas na comunicação.",
                "expected": "Alto (61-85%)"
            },
            
            "Teste Crítico": {
                "text": "Cliente extremamente revoltado. Quarto contato do dia! Disse que vai acionar o Procon e processar a empresa. Defeito persiste e está causando prejuízo financeiro. Situação inaceitável!",
                "expected": "Crítico (86-100%)"
            }
        }
        
        selected_example = st.selectbox("Escolha um exemplo de teste:", list(examples.keys()))
        
        st.write(f"**Texto do teste:** {examples[selected_example]['text']}")
        st.write(f"**Resultado esperado:** {examples[selected_example]['expected']}")
        
        if st.button("🧪 EXECUTAR TESTE", key="test_example"):
            st.write(f"**🔬 Testando: {selected_example}**")
            st.markdown("---")
            analyze_text_with_exact_prompt(
                analyzer, 
                examples[selected_example]["text"], 
                "Exemplo de Teste", 
                f"TESTE_{selected_example.replace(' ', '_').upper()}"
            )

if __name__ == "__main__":
    main()
