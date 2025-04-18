from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Carregar as chaves APIs presentes no arquivo .env
load_dotenv()
# --------------------------------------------------------------------------------

# Instanciar um chatmodel para comunicarmos com os modelos LLMs
model_classificador = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------------------------------------------------------------
# Criando o classificador da pergunta de entrada do usuário:
class ClassificaEntrada(BaseModel):
    opcao: int = Field(description="Defina 1 se a pergunta do usuário for referente à dúvidas gerais de um FAQ. \
Defina 2 se for uma solicitação de ajuda para montar um treino ou pergunta específica sobre um exercicio ou treino. \
defina 3 se for saudações ou temas que não são referentes à dúvidas sobre academia.")

# Criando o parser estruturado
parser_classifica = PydanticOutputParser(pydantic_object=ClassificaEntrada)


# Criando o ChatPromptTemplate que solicitará ao LLM que ele classifique a entrada do usuário:
sys_prompt_rota = """Você é um especialista em classificação. Você receberá perguntas do usuário e precisará classificar, \
de forma booleana, se o usuário está perguntando sobre dúvidas gerais sobre a academia e planos ou se ele precisa \
de ajuda com um treino ou exercício.
\n{format_instructions}\n
Pergunta Usuário: {input}"
"""

rota_prompt_template = ChatPromptTemplate([("system", sys_prompt_rota),],
                                          partial_variables={"format_instructions": parser_classifica.get_format_instructions()}
                                          )

# Criando a Chain que vai classificar a entrada do usuário:
chain_de_roteamento = rota_prompt_template | model_classificador | parser_classifica
print(chain_de_roteamento)
