from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

@tool
def traducao(str):
    '''
    traduza todo o texto antes de responder, em portugues do brasil'''
    return str

@tool
def cotacao_dolar(dinheiro) -> float:
    """Consulta a cotação atual do dinheiro em reais usando a AwesomeAPI.
    argumento exemplo de moeda BTC"""
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
    """Converte um valor em dinheiro e dividi
    args cotaçao"""
    
    return float(valor_dinheiro) / 10


tools = [cotacao_dolar, calcular_divisor, traducao]

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")
llm2 = ChatOpenAI(temperature=0.1, model_name="gpt-4o")



agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

pergunta = "quero a cotaçao de 4 moedas pode ser qualquer uma"
resposta = agent.invoke(pergunta)
print('------------------------------------------')
# print(resposta)

template = '''
Voce é um agente financeiro especialista em cotaçoes
contexto: {output}
pergunta do usuario:{input},
''' 

prompt = PromptTemplate(template=template, input_variables=["input", "output"])

chain = prompt | llm2 | StrOutputParser()


ckunks = []
for chunk in chain.stream({'input': resposta.get('input'), 'output': resposta.get('output')}):
    ckunks.append(chunk)
    print(chunk, end="", flush=True)


