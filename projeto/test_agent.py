import streamlit as st
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuração inicial do Streamlit
st.set_page_config(page_title="Agente Financeiro", page_icon="💰")
st.title("Agente Financeiro Inteligente")

# Carrega variáveis de ambiente
load_dotenv()

# Define as ferramentas do agente
@tool
def traducao(texto: str) -> str:
    '''Traduz todo o texto para português do Brasil antes de responder'''
    return texto

@tool
def cotacao_moeda(moeda: str) -> float:
    """
    Obtém a cotação atual de uma moeda em reais brasileiros usando a AwesomeAPI.

    Parâmetros:
    moeda (str): O código da moeda (por exemplo, BTC para Bitcoin).

    Retorna:
    float: A cotação atual da moeda em reais brasileiros.
    """
    try:
        url = f"https://economia.awesomeapi.com.br/json/last/{moeda}-BRL"
        response = requests.get(url)
        data = response.json()
        cotacao = float(data[f"{moeda}BRL"]["bid"])
        return cotacao
    except Exception as e:
        return f"Erro ao consultar cotação: {e}"

@tool
def calcular_divisor(valor_dinheiro: float) -> float:
    """Divide um valor em dinheiro por 10"""
    return float(valor_dinheiro) / 10

# Inicializa o agente
def inicializar_agente():
    tools = [cotacao_moeda, calcular_divisor, traducao]
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

# Template para formatação da resposta
template = '''
Você é um agente financeiro especialista em dinheiro.
s
Contexto:
{output}

Pergunta do usuário: {input}'''

prompt = PromptTemplate(template=template, input_variables=["input", "output"])
chain = prompt | ChatOpenAI(temperature=0.1, model_name="gpt-4o") | StrOutputParser()

# Interface do Streamlit
pergunta = st.text_input("Faça sua pergunta financeira (ex: qual a cotação do dólar e use o divisor para mim):")

if st.button("Consultar") or pergunta:
    with st.spinner("Processando sua consulta..."):
        try:
            # Inicializa e executa o agente
            agent = inicializar_agente()
            resposta = agent.invoke(pergunta)
            
            # Formata a resposta
            resposta_formatada = []
            for chunk in chain.stream({'input': resposta.get('input'), 'output': resposta.get('output')}):
                resposta_formatada.append(chunk)
            
            # Exibe os resultados
            st.subheader("Resposta:")
            st.write("".join(resposta_formatada))
            
            st.subheader("Detalhes da Execução:")
            st.json(resposta)  # Mostra a resposta completa do agente
            
        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")

# Seção de informações
st.sidebar.markdown("""
## Funcionalidades do Agente

- Consulta cotações de moedas (USD, BTC, etc)
- Realiza cálculos financeiros
- Traduz respostas para português
- Explica conceitos financeiros

Exemplos de perguntas:
- Qual a cotação do dólar hoje?
- Quanto é 100 dólares divididos por 10?
- Me explique sobre Bitcoin em português
""")