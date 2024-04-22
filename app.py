import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
import google.generativeai as genai

# Load .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Check your .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title('Chat with Your Own Data')

def init_chat_engine(index, llm_model, memory): 
    return index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm_model,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
)

if 'messages' not in st.session_state: 
    st.session_state['messages'] = []

with st.sidebar: 
    data_dir = st.text_input("Enter your data directory", value="data") 
    if st.button('Ingest and Index Data'): 
        if data_dir: 
            with st.spinner('Ingesting and indexing data, please wait...'): 
                try: # Ingest and index the data 
                    documents = SimpleDirectoryReader(data_dir).load_data() 
                    llm_model = Gemini(models='models/gemini-1.5-pro-latest', api_key=GOOGLE_API_KEY) 
                    gemini_embed_model = GeminiEmbedding( model_name="models/embedding-001", api_key=GOOGLE_API_KEY, title="this is a document" )

                    # Update settings
                    settings.llm = llm_model
                    settings.embed_model = gemini_embed_model
                    settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
                    settings.num_output = 512
                    settings.context_window = 3900

                    # Indexing
                    index = VectorStoreIndex.from_documents(documents, embed_model=gemini_embed_model)
                    index.storage_context.persist()

                    # Initialize chat engine
                    chat_engine = init_chat_engine(index, llm_model, ChatMemoryBuffer(token_limit=3900))

                    # Set session state
                    st.session_state['llm_model'] = llm_model
                    st.session_state['index'] = index
                    st.session_state['chat_engine'] = chat_engine

                    st.success('Data ingested and indexed successfully!')

                except Exception as e:
                    st.error(f'An error occurred during ingestion and indexing: {e}')

# Check if indexing was successful
if 'index' in st.session_state and st.session_state['index'] is not None:
    
    chat_engine = st.session_state['chat_engine']

    ##init session state
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your data."}
        ]

    assistant = "\U0001F916"
    user = "\U0001F97A"
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # if user enters something
    if prompt := st.chat_input("Ask the AI Assistant a question for your data"):
        response = chat_engine.chat(prompt) 

        # Display user message in chat message container
        st.chat_message("user", avatar=None).markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=None):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar: 
        if st.button('Reset chat memory'):
            chat_engine.reset()
        



