# SRO Risk Analyzer - Sistema de Análise Preditiva de Reclamações

## 📋 Descrição

O **SRO Risk Analyzer** é uma aplicação web desenvolvida com Streamlit que utiliza Inteligência Artificial para prever a probabilidade de clientes formalizarem reclamações (SRO - Sistema de Registro de Ocorrências) com base em registros de atendimento.

O sistema implementa uma **metodologia estruturada de 4 fatores ponderados** para análise preditiva, combinando regras baseadas em palavras-chave com análise avançada usando GPT-4.

## 🎯 Funcionalidades Principais

### 📊 **Análise Preditiva Estruturada**
- **4 Fatores Ponderados** conforme metodologia especializada
- **Classificação de Risco**: Baixo, Médio, Alto, Crítico
- **Score Percentual**: 0% a 100% de probabilidade de reclamação
- **Análise Híbrida**: Combinação de regras locais + GPT-4

### 📈 **Visualizações Interativas**
- **Gauge de Risco**: Indicador visual do nível de risco
- **Gráfico de Fatores**: Breakdown detalhado por categoria
- **Métricas em Tempo Real**: Scores e classificações

### 🧪 **Sistema de Testes**
- **Exemplos Pré-definidos**: Para cada nível de risco
- **Testes Customizados**: Crie seus próprios exemplos
- **Validação da Metodologia**: Teste e refine a análise

## 🔬 Metodologia de Análise

### **Fatores Ponderados:**

1. **🔢 Frequência de Contatos (Peso 4)**
   - 1 contato: risco baixo
   - 2 contatos: risco médio
   - 3+ contatos: risco elevado
   - *Atenuação*: Palavras neutras reduzem o risco

2. **⏰ Tempo de Espera (Peso 3)**
   - Detecção de atrasos e urgência
   - Padrões de insatisfação temporal
   - Tolerâncias por tipo de serviço

3. **⚙️ Falhas Operacionais (Peso 2)**
   - **Indícios técnicos**: defeitos, vazamentos, quebras
   - **Falhas de processo**: cadastro, comunicação, pós-serviço

4. **😠 Estado Emocional (Peso 1)**
   - **Negativos moderados**: frustrado, decepcionado (+1 ponto)
   - **Risco jurídico**: Procon, processar, advogado (+3 pontos)
   - **Positivos**: excelente, satisfeito (-1 ponto)

### **Classificação de Risco:**
- **Baixo**: 0-30%
- **Médio**: 31-60%
- **Alto**: 61-85%
- **Crítico**: 86-100%

## 🚀 Como Configurar e Executar

### **Pré-requisitos**
- Python 3.8 ou superior
- Conta OpenAI com API Key

### **Instalação**

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/sro-risk-analyzer.git
   cd sro-risk-analyzer
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure a API Key da OpenAI:**

   **Para Streamlit Cloud:**
   - Acesse Settings > Secrets no seu app
   - Adicione:
     ```toml
     OPENAI_API_KEY = "sua-chave-openai-aqui"
     ```

   **Para execução local:**
   - Crie `.streamlit/secrets.toml`:
     ```toml
     OPENAI_API_KEY = "sua-chave-openai-aqui"
     ```

4. **Execute a aplicação:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Estrutura do Projeto

```
sro-risk-analyzer/
├── streamlit_app.py          # Aplicação principal
├── requirements.txt          # Dependências Python
├── README.md                # Documentação
├── Dados_SRO.pkl           # Base histórica (para versão futura com embeddings)
├── dados_semSRO.pkl        # Base complementar (reservado)
└── .streamlit/
    └── secrets.toml         # Configurações locais (não commitado)
```

## 🎮 Como Usar

### **1. 📤 Upload de Arquivo**
- Formatos suportados: PDF, Excel, JSON, TXT
- Extração automática de texto
- Análise de documentos estruturados

### **2. ✍️ Texto Manual**
- Cole diretamente registros de atendimento
- Análise em tempo real
- Ideal para testes rápidos

### **3. 🧪 Exemplos de Teste**
- **Exemplos pré-definidos** para cada nível de risco
- **Testes customizados** para validar a metodologia
- **Validação rápida** do sistema

## 📊 Exemplo de Saída

```
- Pedido: ORD123456
- Probabilidade de Reclamação: Crítica
- Porcentagem Estimada: 92%
- Fatores Críticos: 4 contatos, ameaça jurídica, problemas técnicos
- Conclusão: Cliente demonstra alta insatisfação e ameaça jurídica. 
  Recomendamos contato imediato com supervisor.
```

## 🔮 Versões e Roadmap

### **Versão Atual: 2.0 - Análise por Prompt**
- ✅ Metodologia estruturada de 4 fatores
- ✅ Análise híbrida (Local + GPT-4)
- ✅ Interface completa com testes
- ✅ Visualizações interativas

### **Versão Futura: 3.0 - RAG + Embeddings**
- 🔄 Integração com base histórica (Dados_SRO.pkl)
- 🔄 Busca por similaridade vetorial
- 🔄 Comparação com casos históricos
- 🔄 Análise ainda mais precisa

## 🛠️ Tecnologias Utilizadas

- **Frontend**: Streamlit
- **IA**: OpenAI GPT-4 + text-embedding-ada-002
- **Visualização**: Plotly
- **Processamento**: Pandas, NumPy
- **Documentos**: PyPDF2, openpyxl

## 📈 Performance e Limites

- **Velocidade**: Análise em ~2-5 segundos
- **Precisão**: Baseada em metodologia especializada
- **Escalabilidade**: Suporta análise individual ou em lote
- **Custo**: ~$0.01-0.03 por análise (via OpenAI API)

## 🤝 Contribuições

1. Faça fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

Para dúvidas, sugestões ou problemas:
- Abra uma [Issue](https://github.com/seu-usuario/sro-risk-analyzer/issues)
- Entre em contato via email: [seu-email@dominio.com]

---

**Desenvolvido com ❤️ para otimização de atendimento ao cliente e prevenção de reclamações**
