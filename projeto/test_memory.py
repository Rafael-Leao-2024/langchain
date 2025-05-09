import streamlit as st
from langchain_openai import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.output_parsers import StrOutputParser  
from langchain_community.chat_message_histories import SQLChatMessageHistory  
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração inicial do Streamlit
st.set_page_config(page_title="Chatbot com Memória", page_icon="🤖")
st.title("Chatbot com Memória de Conversa")

# Inicializa a sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Configuração do prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Atue como um assistente de IA útil"),
    MessagesPlaceholder(variable_name='history'),
    ("human", "{human_input}")
])

# Persistir conversas em banco de dados SQL
def get_session_history_db(session_id):  
    return SQLChatMessageHistory(session_id, connection="sqlite:///memory.db") 

# Função de janela de buffer de memória
def memory_window(messages, k=10):  
    return messages[-(k+1):]

# Cadeia LLM (pode alternar entre Groq e OpenAI)
# Escolha o modelo que preferir (descomente um):
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

llm_chain = (
    RunnablePassthrough.assign(history=lambda x: memory_window(x["history"]))  
    | prompt  
    | llm  
    | StrOutputParser()
)

# Cadeia de conversação com histórico
conv_chain = RunnableWithMessageHistory(
    llm_chain,
    get_session_history_db,
    input_messages_key="human_input",
    history_messages_key="history"
)

# Lógica do chat
if input_prompt := st.chat_input("Digite sua mensagem..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("user"):
        st.markdown(input_prompt)
    
    # Obtém resposta do chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Usa session_id baseado no usuário do Streamlit para persistência
        session_id = str(hash(st.user.__str__()))
        
        # Invoca a cadeia de conversação
        response = conv_chain.invoke(
            input={"human_input": input_prompt},
            config={'configurable': {'session_id': session_id}}
        )
        
        message_placeholder.markdown(response)
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})