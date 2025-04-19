from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests


@tool
def cotacao_dolar(dinheiro) -> float:
    """Consulta a cotação atual do dinheiro em reais usando a AwesomeAPI.
    argumento exemplo de moeda ABC"""
    try:
        url = f"https://economia.awesomeapi.com.br/json/last/{dinheiro}-BRL"
        response = requests.get(url)
        data = response.json()
        cotacao = float(data[f"{dinheiro}BRL"]["bid"])
        return cotacao
    except Exception as e:
        return f"Erro ao consultar cotação: {e}"


@tool
def calcular_divisor(valor_dinheiro: float) -> float:
    """Converte um valor em dinheoiro por 2 
    args cotaçao"""
    
    return float(valor_dinheiro) / 2


tools = [cotacao_dolar, calcular_divisor]

llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

for pedaco in agent.stream("quero saber a cotacao de uma moeda"):
    print(pedaco)
