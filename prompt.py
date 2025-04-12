from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage

print("=========== Exemplo 3 - Alternativa 1 =========")


prompt_template = ChatPromptTemplate([
    ("system", "Você é um assistente de IA com habilidade de escritor de poesia."),
	MessagesPlaceholder("msgs_user")
]
)

retorno = prompt_template.invoke(
    {"msgs_user": [HumanMessage(content="Gere para mim um poema sobre: navegação. Escreva em pt-br")]
     })

for role, content in retorno:
    print(f"{role.capitalize()}: {content}")
    break

