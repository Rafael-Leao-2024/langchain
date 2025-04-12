from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


from dotenv import load_dotenv

load_dotenv()


# Define a sessão do Streamlit
st.set_page_config(page_title="Jasmine - Assistente de Documentos", layout="wide")
st.title("📄 Jasmine - Despachante Inteligente")
st.markdown("Faça uma pergunta com base no conteúdo do PDF. Jasmine te responde na hora!")


# Carrega e indexa o PDF apenas uma vez por sessão
@st.cache_resource(show_spinner="Carregando e indexando o PDF...")
def carregar_vectorstore():
    caminho_pdf = "documento_informacoes.pdf"

    # Lendo arquivo pdf
    loader = PyPDFLoader(caminho_pdf)
    pages = [page for page in loader.load()] 

    # splitando os documentos 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n"], is_separator_regex=False)
    documents =  splitter.split_documents(pages)

    # indexando vectorstore e fazendo o embedding
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    return vectorstore


vectorstore = carregar_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Definição do template
template = """
Você é Jasmine,  se indetifique se perguntarem seu nome, uma despachante experiente, clara e eficiente. Responda à pergunta do usuário com base apenas nas informações fornecidas no contexto abaixo. 
Se a resposta não estiver presente a base de informaçao da Recife placas, diga educadamente que não é possível responder com os dados disponíveis, examine bem o contexto e responda.

Contexto:
{context}

Pergunta:
{question}

Resposta (por Jasmine):
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question", "data"])
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
parser = StrOutputParser()

chain = prompt | llm | parser

with st.form("form_pergunta"):
    pergunta = st.text_input("Digite sua pergunta sobre o documento:", placeholder="Ex: Qual o país do endereço desse recibo?")
    enviar = st.form_submit_button("Perguntar")

if enviar and pergunta:
    contextos = retriever.invoke(pergunta)
    contexto = "\n".join(ctx.page_content for ctx in contextos)

    st.markdown("### ✨ Resposta da Jasmine:")
    resposta_area = st.empty()

    # Exibir resposta com streaming
    resposta_final = ""
    try:
        for chunk in chain.stream({"context": contexto, "question": pergunta, "data": data}):
            resposta_final += chunk
            #print(resposta_final)
            resposta_area.markdown(f"🪪 {resposta_final}▌")

        resposta_area.markdown(f"🪪 {resposta_final}")  # Remove o cursor final
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")

