import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import streamlit as st
from query_mapping import query_type_mapping,instructions
import boto3

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY and OPENAI_API_KEY in your .env file")

def load_player_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['players']

def get_text_chunks(players: List[Dict], chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    
    for player in players:
        player_text = json.dumps(player)
        chunks = text_splitter.split_text(player_text)
        all_chunks.extend(chunks)
    
    return all_chunks
def get_local_vectors(faiss_index_path):
    if os.path.exists(faiss_index_path):
        print("Loading FAISS index from faiss_index.faiss...")
        vector_store = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        return vector_store

def get_vector_store(text_chunks):
    faiss_index_path = "faiss_index.faiss"
    if os.path.exists(faiss_index_path):
        print("Loading FAISS index from faiss_index.faiss...")
        vector_store = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating a new FAISS index...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(faiss_index_path)
    
    return vector_store
def load_prompt_template(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def get_query_type(question):
    classification_prompt = load_prompt_template("classification_prompt.txt")
    classification_prompt = classification_prompt.format(question=question)
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    response = client.invoke_model(
        modelId='anthropic.claude-v2',
        body={"inputText": classification_prompt},
        contentType='application/json'
    )
    
    response_content = response['body']['completions'][0]['completion'].strip().lower()

    valid_categories = ["vct_international", "vct_challengers", "game_changers", "mixed_gender", "cross_regional", "rising_star", "general"]

    if response_content in valid_categories:
        return response_content
    else:
        print(f"Warning: Unexpected classification '{response_content}'. Defaulting to 'general'.")
        return "general"


def load_prompt_template(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def get_prompt_template(query_type):
    specific_instructions = instructions.get(query_type, instructions["general"])
    base_prompt = load_prompt_template("valorant_prompt.txt")
    return base_prompt.format(specific_instructions=specific_instructions)

def get_conversational_chain(query_type):
    base_prompt = get_prompt_template(query_type)
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=base_prompt, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def create_team_composition(vector_store, chain, question):
    docs = vector_store.similarity_search(question, k=10)
    context = "\n".join([doc.page_content for doc in docs])
    context_doc = Document(page_content=context)
    response = chain.invoke({"input_documents": [context_doc], "question": question})
    return response["output_text"]

def main(question):
    faiss_index_path = "faiss_index.faiss"
    vector_store=get_local_vectors(faiss_index_path)
    query_type = get_query_type(question)
    question=query_type_mapping.get(query_type,query_type_mapping["general"])
    chain = get_conversational_chain(query_type)
    team_composition = create_team_composition(vector_store, chain, question)
    return team_composition[8:-4]
  
def save_json_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
if __name__ == "__main__":
    question = input("Enter your question: ")
    result = main(question)
    result=json.loads(result)
    save_json_data("teamdetail.json",result)
    print(result)