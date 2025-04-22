from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders.sql_database import SQLDatabaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


# 1. Conectando ao banco PostgreSQL
db_url = "postgresql://postgres:lzWXMUiEnpheEfywuljmmelShahBQyIm@junction.proxy.rlwy.net:35789/railway"
db = SQLDatabase.from_uri(db_url)

# 2. Pegando todas as tabelas disponíveis
tabelas = db.get_usable_table_names()
# print("Tabelas encontradas:", tabelas)

# 3. Carregando os dados de todas as tabelas
docs = []
for tabela in tabelas:
    if tabela != "alembic_version":
        query = f"SELECT * FROM {tabela}"
        loader = SQLDatabaseLoader(query=query, db=db)
        docs_tabela = loader.load()
        docs.extend(docs_tabela)

# print(f"Total de documentos carregados: {len(docs)}")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len, separators=["\n"], is_separator_regex=False)
documents_splits =  splitter.split_documents(docs)
# 4. Embeddings + FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents_splits, embeddings)

# 5. Salvando o índice FAISS localmente (opcional)
# vectorstore.save_local("faiss_index_todas_tabelas")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# Template com contexto vindo do banco
template = """
Você é um assistente que responde perguntas com base nos dados abaixo extraídos do banco:

{context}

Pergunta: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

pergunta = 'qual email de rafael:'
contextos = retriever.invoke(pergunta)
contexto = "\n".join((f"Source: {ctx.metadata}\n\n" f"Content: {ctx.page_content}") for ctx in contextos)
# Monta a cadeia completa
llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o")
chain = (
    prompt
    | llm
)
print(contextos)
print(contexto)
for chunk in chain.stream({"context": contexto, "question": pergunta}):
    print(chunk.content, end="", flush=True)
