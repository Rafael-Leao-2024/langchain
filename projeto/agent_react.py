from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.tools import tool as ferramenta

load_dotenv()

@ferramenta
def vale_apena(cotacao):
    '''
    recebi como entrada a cotacao vinda da cotaçao_moeda
    argumento float
    '''
    if float(cotacao) >= 5.7283:
        return f"Nao vale a pena ta caro com valor de {cotacao}"
    else:
        return f"Vale a Pena cotacao {cotacao}"

@tool
def traducao(str):
    '''
    traduza todo o texto antes de responder, em portugues do brasil'''
    
    return str


@tool
def cotaçao_moeda(dinheiro) -> float:
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


tools = [cotaçao_moeda, calcular_divisor, traducao, vale_apena]

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")
llm2 = ChatOpenAI(temperature=0.1, model_name="gpt-4o")



agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

pergunta = "vale a pena compra dolar? e se ta caro quanto ta a mais ?"  
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

entrada_chain = {'input': resposta.get('input'), 'output': resposta.get('output')}

ckunks = []
for chunk in chain.stream(entrada_chain):
    ckunks.append(chunk)
    print(chunk, end="", flush=True)


