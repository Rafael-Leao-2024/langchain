from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


async def conversa_com_modelo():
    model = ChatOpenAI(model="gpt-4o", temperature=0.1)
    conversa = [SystemMessage(content="Você é um assistente útil que responde ao usuário com detalhes e exemplos.")]

    while True:
        entrada = input("\nEntrada Usuário (digite 'q' para parar.): ")
        if entrada.lower() == "q":
            break

        conversa.append(HumanMessage(content=entrada))

        # Ajustado aqui:
        all_chunk = []
        async for chunk in model.astream(conversa):
            all_chunk.append(chunk.content)
            print(chunk.content, end="", flush=True)

            # Para armazenar na conversa a resposta do modelo ou caso você queira ver a resposta final você precisará
        # juntar os chunks armazenados em `all_chunk` antes:
        resposta_completa = "".join(all_chunk)
        conversa.append(AIMessage(content=resposta_completa))
        # print(f"\nResposta final: {resposta_completa}")

    print("---- Histórico Completo ----")
    print(conversa)


import asyncio  # Biblioteca para lidar com funções assíncronas

asyncio.run(conversa_com_modelo())