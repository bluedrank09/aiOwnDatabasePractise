# importing libraries
import gradio as gr # gradio library
import sys
import os # most likely to be used to get api key ? 
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

def construct_index(directory_path): # contructing nodes and grouping them to use later for KNN. we need drectry pat to get to te pdfs
    max_input_size = 4096 # bytes ? pixels ? characters? check
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit = chunk_size_limit)
    llm_predictor = LLMPredictor(llm = ChatOpenAI(temperature = 0.7, model_name = 'gpt-3.5-turbo', max_tokens = num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data
    index = GPTSimpleVectorIndex(documents, llm_predictor = llm_predictor, prompt_helper = prompt_helper)

    index.save_to_disk("index.json") 

    return(index)

def chatBot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk("index.json")
    response = index.query(input_text, response_mode = 'compact')
    
    return(response.response)

iFace = gr.Interface(fn = chatBot, inputs = gr.components.Textbox(lines = 7, label = "Enter your text here : "), outputs = 'text', title = "Test for custom AI Chatbot")

index = construct_index("docs")
iFace.launch(share = True)
