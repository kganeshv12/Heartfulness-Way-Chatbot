import openai, os

openai.api_key = os.environ["OPENAI_API_KEY"]
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files = ["./HFNWAY.pdf"]
).load_data()

from llama_index.core.schema import Document
document = Document(text = "\n\n".join([doc.text for doc in documents]))
# above step merges it the 218 different documents into 1 document

from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI

# resp = OpenAI().complete("Lord Ram is ")
# print(resp)
# does not need api key
# the embedding model we use is hugging face  bge small
# the service context class already hass this

llm = OpenAI(model= "gpt-3.5-turbo", temperature=0.2)
service_context = ServiceContext.from_defaults(
    llm = llm, embed_model = "local:BAAI/bge-small-en-v1.5"
)
# https://huggingface.co/BAAI/bge-small-en
index = VectorStoreIndex.from_documents([document], service_context=service_context)
# pip install llama-index-embeddings-huggingface
# this cell takes care of chunking embedding, indexing

query_engine = index.as_query_engine()

import gradio as gr

def chat_interface(message, history):
  
  # Update history with the current message
  # Call the query function with the user's message
  response = str(query_engine.query(message))
  # Update history with the bot's response
  
  history.append([message, response])
  return response


# Initialize an empty history
history = []
# Create the chat interface using gr.ChatInterface
interface = gr.ChatInterface(chat_interface, examples = ["Who is DAAJI?", "What is Heartfulness way of meditation all about?"] , title = "HFN BOT")
# Launch the interface
interface.launch()


