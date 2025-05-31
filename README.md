# Projeto SRO: Sistema de Previsão de Reclamações com IA e RAG

## Descrição

O Sistema de Previsão de Reclamações (SRO) é uma aplicação web desenvolvida com Streamlit que utiliza técnicas avançadas de Inteligência Artificial, especificamente Retrieval Augmented Generation (RAG), para prever a probabilidade de reclamações formais com base em comentários de atendimento ao cliente.

A aplicação analisa comentários de atendimento, compara-os com uma base histórica de 36 mil registros que resultaram em reclamações, e fornece uma análise preditiva detalhada sobre o risco de uma reclamação formal ser aberta, permitindo ações preventivas.

## Tecnologias Utilizadas

- **Streamlit**: Framework para interface web
- **OpenAI API**: Modelos GPT-4 para análise e text-embedding-ada-002 para embeddings
- **RAG (Retrieval Augmented Generation)**: Técnica que combina recuperação de informações com geração de texto
- **FAISS**: Biblioteca para busca eficiente de similaridade vetorial
- **LangChain**: Para processamento e chunking de texto
- **Pandas/NumPy**: Para manipulação de dados
- **PyMuPDF**: Para processamento de arquivos PDF

## Como Configurar e Rodar

### Pré-requisitos

- Python 3.8 ou superior
- Acesso à API da OpenAI

### Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/sro-previsao-reclamacoes.git
   cd sro-previsao-reclamacoes
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Configure a chave da API OpenAI:
   
   **Para Streamlit Cloud:**
   - Acesse as configurações do seu aplicativo no Streamlit Cloud
   - Vá para "Secrets"
   - Adicione sua chave da API no formato:
     ```
     OPENAI_API_KEY = "sua-chave-aqui"
     ```

   **Para execução local:**
   - Crie um arquivo `.streamlit/secrets.toml` com o conteúdo:
     ```
     OPENAI_API_KEY = "sua-chave-aqui"
     ```

4. Posicionamento do Arquivo Histórico:
   - O arquivo `InformaçõesSRO.xlsx - Planila3.csv` deve estar no **mesmo diretório** do `streamlit_app.py`
   - Este arquivo contém a base histórica de comentários que resultaram em reclamações

5. Execute a aplicação:
   ```
   streamlit run streamlit_app.py
   ```

## Lógica RAG Explicada

O sistema utiliza a técnica RAG (Retrieval Augmented Generation) para melhorar significativamente a precisão das previsões:

1. **Processamento da Base Histórica**:
   - Na primeira execução, o sistema carrega a base de 36 mil comentários históricos
   - Cada comentário é dividido em chunks (pedaços) de 500 caracteres com 100 caracteres de sobreposição
   - Para cada chunk, é gerado um embedding (representação vetorial) usando a API da OpenAI
   - Os embeddings são indexados usando FAISS para permitir busca eficiente por similaridade
   - O índice FAISS e os metadados dos chunks são salvos em disco para carregamento rápido em execuções futuras

2. **Análise de Novos Comentários**:
   - Quando um novo comentário é enviado para análise, o sistema gera seu embedding
   - Usando o índice FAISS, o sistema encontra os 3-5 comentários históricos mais similares
   - Estes comentários similares, junto com o novo comentário, são enviados para o GPT-4
   - O modelo considera fatores como frequência de contatos, tempo de espera, falhas processuais e estado emocional do cliente
   - A análise também considera palavras-chave frequentemente associadas a reclamações

3. **Persistência e Performance**:
   - Após a primeira execução (que pode levar alguns minutos), as execuções subsequentes são muito mais rápidas
   - O sistema carrega o índice FAISS e os metadados diretamente do disco, evitando reprocessamento
   - A função de carregamento é decorada com `@st.cache_resource` para otimizar o uso de memória

## Uso da Aplicação

1. **Upload de Arquivo**:
   - Faça upload de um arquivo Excel, CSV, PDF ou JSON contendo comentários de atendimento
   - Para arquivos Excel/CSV, selecione as colunas que contêm o ID do pedido e o comentário
   - Para PDFs e JSONs, o sistema tentará extrair automaticamente os comentários

2. **Análise**:
   - Clique no botão "Analisar Comentários" para iniciar o processamento
   - O sistema processará cada comentário, encontrará exemplos históricos similares e gerará uma análise

3. **Resultados**:
   - Os resultados são exibidos em formato visual com código de cores por nível de risco
   - Cada análise inclui:
     - Probabilidade de reclamação (Baixa, Média, Alta, Crítica)
     - Porcentagem específica de risco
     - Fatores críticos identificados
     - Conclusão com recomendação de ação preventiva

4. **Download**:
   - Baixe um relatório Excel completo com todos os resultados para análise offline

## Formato de Entrada/Saída

### Entrada:
- Arquivos Excel, CSV, PDF ou JSON contendo comentários de atendimento
- Para Excel/CSV, o usuário pode selecionar as colunas relevantes

### Saída:
- Análise visual na interface com código de cores por nível de risco
- Relatório detalhado em formato Excel para download
- Cada análise inclui probabilidade, porcentagem, fatores críticos e conclusão

## Considerações de Performance

- A primeira execução será mais lenta devido à necessidade de processar a base histórica completa
- Execuções subsequentes serão significativamente mais rápidas graças ao carregamento do índice do disco
- O sistema foi otimizado para lidar com a base de 36 mil registros, mas pode requerer mais memória para bases maiores
