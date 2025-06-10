import streamlit as st
import pandas as pd
import openai
import json
import PyPDF2
from datetime import datetime
import re

# Configuração da página
st.set_page_config(
    page_title="SRO Risk Analyzer - Teste do Prompt PDF",
    page_icon="🔍",
    layout="wide"
)

def test_prompt_from_pdf(client, text, order_id="TESTE"):
    """Testa EXATAMENTE o prompt do PDF anexado"""
    
    # PROMPT EXATO DO PDF ANEXADO - TEXTO LITERAL
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

    # Fazer chamada para GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um especialista em análise preditiva de reclamações seguindo metodologia específica."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()

def extract_text_from_file(uploaded_file):
    """Extrai texto de arquivos"""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(uploaded_file, header=None)
        text_parts = []
        for index, row in df.iterrows():
            for cell_value in row:
                if pd.notna(cell_value) and str(cell_value).strip():
                    text_parts.append(str(cell_value).strip())
        return " | ".join(text_parts)
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    else:
        return "Tipo de arquivo não suportado"

def main():
    st.title("🧪 SRO Risk Analyzer - TESTE DO PROMPT PDF")
    st.markdown("**Objetivo: Testar se o prompt do PDF anexado está funcional**")
    
    st.warning("""
    ⚠️ **ESTE É UM TESTE DO PROMPT**
    
    Esta versão usa o texto LITERAL e COMPLETO do PDF como prompt para verificar se a metodologia funciona.
    """)
    
    # Verificar API Key
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("🔑 API Key carregada")
        client = openai.OpenAI(api_key=api_key)
    except:
        st.error("🔑 API Key não encontrada nos secrets")
        st.stop()
    
    # Sidebar com informações
    with st.sidebar:
        st.header("📋 Metodologia do PDF")
        st.info("""
        **4 Fatores Ponderados:**
        1. Frequência Contatos (Peso 4)
        2. Tempo de Espera (Peso 3)
        3. Falhas Operacionais (Peso 2)
        4. Estado Emocional (Peso 1)
        
        **Classificações:**
        • Baixo: 0-30%
        • Médio: 31-60%
        • Alto: 61-85%
        • Crítico: 86-100%
        """)
    
    # Interface principal
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "✍️ Texto", "🧪 Exemplos"])
    
    with tab1:
        st.header("📤 Teste com Arquivo")
        uploaded_file = st.file_uploader("Escolha um arquivo", type=['pdf', 'xlsx', 'xls', 'txt'])
        order_id = st.text_input("ID do Pedido", value="ARQUIVO_TESTE")
        
        if uploaded_file and st.button("🧪 TESTAR PROMPT"):
            with st.spinner("Extraindo texto..."):
                text = extract_text_from_file(uploaded_file)
            
            st.subheader("📄 Texto Extraído")
            st.text_area("", text[:1000] + "..." if len(text) > 1000 else text, height=150)
            
            with st.spinner("🤖 Testando prompt do PDF..."):
                try:
                    result = test_prompt_from_pdf(client, text, order_id)
                    
                    st.success("✅ PROMPT EXECUTADO COM SUCESSO!")
                    st.subheader("🤖 Resposta do GPT-4o (Prompt do PDF)")
                    st.code(result, language="text")
                    
                    # Download do resultado
                    test_data = {
                        "timestamp": datetime.now().isoformat(),
                        "arquivo": uploaded_file.name,
                        "texto_analisado": text,
                        "resposta_gpt": result
                    }
                    
                    st.download_button(
                        "📥 Baixar Resultado",
                        json.dumps(test_data, ensure_ascii=False, indent=2),
                        f"teste_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Erro ao executar prompt: {str(e)}")
    
    with tab2:
        st.header("✍️ Teste com Texto Manual")
        order_id = st.text_input("ID do Pedido", value="MANUAL_TESTE", key="manual_id")
        text = st.text_area("Digite o texto para análise:", height=200)
        
        if text and st.button("🧪 TESTAR PROMPT", key="manual_test"):
            with st.spinner("🤖 Testando prompt do PDF..."):
                try:
                    result = test_prompt_from_pdf(client, text, order_id)
                    
                    st.success("✅ PROMPT EXECUTADO COM SUCESSO!")
                    st.subheader("🤖 Resposta do GPT-4o (Prompt do PDF)")
                    st.code(result, language="text")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao executar prompt: {str(e)}")
    
    with tab3:
        st.header("🧪 Exemplos de Teste")
        
        examples = {
            "Baixo Risco": "Cliente agradeceu pelo excelente atendimento. Serviço foi perfeito e rápido. Muito satisfeito.",
            "Médio Risco": "Cliente ligou duas vezes. Está com pressa. Demonstrou frustração com demora.",
            "Alto Risco": "Terceiro contato. Defeito voltou. Cliente muito decepcionado e revoltado com situação.",
            "Crítico": "Quarto contato! Cliente vai acionar Procon. Ameaça processar empresa. Prejuízo inaceitável!"
        }
        
        example_name = st.selectbox("Escolha um exemplo:", list(examples.keys()))
        st.write(f"**Texto:** {examples[example_name]}")
        
        if st.button("🧪 TESTAR EXEMPLO", key="example_test"):
            with st.spinner("🤖 Testando prompt do PDF..."):
                try:
                    result = test_prompt_from_pdf(client, examples[example_name], f"EXEMPLO_{example_name.upper()}")
                    
                    st.success("✅ PROMPT EXECUTADO COM SUCESSO!")
                    st.subheader(f"🤖 Resultado para: {example_name}")
                    st.code(result, language="text")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao executar prompt: {str(e)}")

if __name__ == "__main__":
    main()
