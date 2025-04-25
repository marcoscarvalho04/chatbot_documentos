
import tempfile 
import os 
import time 
import streamlit as st
import torch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv 

load_dotenv()
#Configurações do streamlit 
st.set_page_config(page_title='Converse com documentos ')
st.title('Converse com documentos ')

model_class = 'openai' #hf_hub, openai, ollama


#st.button('Botão')
#st.chat_input('Digite sua mensagem')

def model_hf_hub(model= 'microsoft/Phi-3-mini-4k-instruct', temperature=0.1):
    return HuggingFaceEndpoint(
        temperature=temperature, 
        repo_id=model,
        return_full_text=False, 
        max_new_tokens=512
    )

def model_openai(model='gpt-4o-mini', temperature=0.1):
    return ChatOpenAI(
        model=model,
        temperature=temperature
    )

def model_ollama(model='phi3', temperature=0.1):
    return ChatOllama(
        model=model,
        temperature=temperature,
    )

def config_retriver(uploads):
    #Carregar os documentos 
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads: 
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
        
        loader = PyPDFLoader(temp_file_path)
        docs.extend(loader.load())

    #Divisão em chunks (split)
    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    splites = text_spliter.split_documents(documents=docs)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

    #Armazenamento no BD de vetor
    vector_store = FAISS.from_documents(splites, embeddings)
    vector_store.save_local('vectorstore/db_faiss')

    #Configuração do retriver 
    retriver = vector_store.as_retriever(search_type='mmr', search_kwargs={
        'k': 3 , 'fetch_k': 7
    })
    return retriver

def config_rag_chain(model_class,retriver):

    # Carregamento da LLM 
    if model_class == 'hf_hub': 
        llm = model_hf_hub()
    elif model_class == 'openai':
        llm = model_openai()
    else:
        llm = model_ollama()

    if model_class.startswith('hf'):
        #user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        token_s, token_e = "<|user|>\n{input}<|end|><|assistant|>", "<|user|>\n{input}<|end|><|assistant|>"
    else: 
        token_s, token_e = "", ""

    #(consulta, historico_chat) -> LLM -> consulta reformulada -> retriver
    context_q_system_prompt = 'Given the following chat history and the follow-up questions which might reference context in chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.'
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = 'Question: {input}' + token_e
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", context_q_user_prompt)
    ])
    #Chain para contextualização  - isso fala sobre a contextulização
    history_aware_retrive = create_history_aware_retriever(llm,
                                                            retriever=retriver,
                                                            prompt=context_q_prompt)
    qa_prompt_template = '''
        Você é um assistente virtual bastante prestativo e está respondendo perguntas gerais. Use os seguintes pedaços de contexto recuperado para
        responder as perguntas. Se não souber a resposta, diga apenas que não sabe a resposta. Mantenha a resposta sempre concisa. 
        Responda em português. \n\n
        Pergunta: {input}\n 
        Contexto: {context}
    '''
    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    #configurar a llm chain para perguntas e respostas 
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retrive, qa_chain)

    return rag_chain




uploads = st.sidebar.file_uploader(
    label='Enviar arquivos',
      type=['PDF'],
      accept_multiple_files=True 
)

if not uploads: 
    st.info('Por favor, envie pelo menos um arquivo para continuar')
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [AIMessage(content='Boas-vindas ao seu assistente! Como posso ajudar?')]

if 'docs_list' not in st.session_state: 
    st.session_state.docs_list = None

if 'retriver' not in st.session_state: 
    st.session_state.retriver = None

for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message('AI'):
            st.write(message.content)

    elif isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.write(message.content)

start = time.time()
user_query = st.chat_input('Digite sua mensagem aqui')

if user_query is not None and user_query != '' and uploads is not None: 

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message('Human'):
        st.markdown(user_query)
    
    with st.chat_message('AI'):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriver = config_retriver(uploads)
        
        rag_chain = config_rag_chain(model_class,st.session_state.retriver)
        
        def content_generator():

            sources = None
            resp = ''
            for chunk in rag_chain.stream({'input': user_query, 'chat_history': st.session_state.chat_history}):
                if chunk.get('answer') is not None and chunk.get('answer') != '':

                    resp = resp+chunk.get('answer')
                    yield chunk.get('answer')

                if sources is None and chunk.get('context') is not None: 
                    sources = chunk.get('context')

            for idx, doc in enumerate(sources): 
                source = doc.metadata['source']
                file = os.path.basename(source)
                page = doc.metadata.get('page', 'Página não especificada')

                #Fonte 1: documento.pdf - p. 2
                ref = f':link: Fonte {idx}: *{file} - p. {page}*'
                with st.popover(ref):
                    st.caption(doc.page_content)

            st.session_state.chat_history.append(AIMessage(content=resp))

        st.write_stream(content_generator())
        #mostrar de onde veio 

    
end = time.time()
print('Tempo: ', end - start)
