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


# Lendo arquivo pdf
caminho_pdf = "servicos_de_veiculos.pdf"
loader = PyPDFLoader(caminho_pdf)
lista_docs = loader.load()
documentos_pages = [page for page in lista_docs]

# Carrega e indexa o PDF apenas uma vez por sessão
@st.cache_resource(show_spinner="Carregando e indexando o PDF...")
def carregar_vectorstore():    

    # splitando os documentos 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len, separators=["\n", "\n\n"], is_separator_regex=False)
    documents_splits =  splitter.split_documents(documentos_pages)

    # indexando vectorstore e fazendo o embedding
    vectorstore = FAISS.from_documents(documents_splits, OpenAIEmbeddings())
    return vectorstore


vectorstore = carregar_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Definição do template
template = """
Você é Jasmine, uma despachante experiente, clara e eficiente. Sempre inicie sua resposta se identificando como Jasmine.

Responda à pergunta do usuário com base **exclusivamente** nas informações fornecidas no contexto abaixo.

📌 Instruções importantes:
- Todas as informações como telefone, endereço ou nome presentes no documento devem ser consideradas como pertencentes à empresa **Grupo Andrade**, salvo indicação explícita em contrário.
- Caso a resposta **não possa ser respondida com o contexto fornecido**, diga isso de forma educada, e em seguida **sugira uma versão melhorada da pergunta**, para que o usuário possa copiá-la e colá-la novamente.

---

📄 Contexto:
{context}

❓ Pergunta:
{question}

---

👩‍💼 Resposta (por Jasmine, com tom humano e profissional):
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") #ler automaticamente a  chave API
parser = StrOutputParser()

chain = prompt | llm | parser # cadeia

with st.form("form_pergunta"):
    pergunta = st.text_input("Digite sua pergunta sobre o documento:", placeholder="Ex: Qual o país do endereço desse recibo?")
    enviar = st.form_submit_button("Perguntar", disabled=False)

if enviar and pergunta:
    contextos = retriever.invoke(pergunta)
    contexto = "\n".join((f"Source: {ctx.metadata}\n\n" f"Content: {ctx.page_content}") for ctx in contextos)
    
    st.markdown("### ✨ Resposta da Jasmine:")
    resposta_area = st.empty()
    # Exibir resposta com streaming
    resposta_final = ""
    try:
        for chunk in chain.stream({"context": contexto, "question": pergunta}):
            resposta_final += chunk
            #print(resposta_final)
            resposta_area.markdown(f"🪪 {resposta_final}▌")
        resposta_area.markdown(f"🪪 {resposta_final}")  # Remove o cursor final

    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
