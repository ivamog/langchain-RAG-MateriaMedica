# This script uses Vector database as a context for answering the question, using LLaMa 2 and Huggingface chatbot.
# Works with RAGforBooks.py which saves vector embeddings to a Chroma database.
#
# Before running the script, you might need to install the following:
# pip install transformers accelerate sentencepiece
#
# Download model from HuggingFace and save the config.json file in the current directory.
# Make sure to have HUGGINGFACE_TOKEN set in the environment variables and request access to the model in HuggingFace.
# curl -H "Authorization: Bearer YOUR_HF_TOKEN" -o config.json https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/config.json
#
# To run the script:
# py RAGPromptOne.py '<Your query text>'
# 

import argparse
import os
from huggingface_hub import login
from dataclasses import dataclass
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # For HuggingFace Embeddings
from langchain_huggingface import HuggingFacePipeline  # For HuggingFace Similarity
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import ChatHuggingFace  # For HuggingFace Chat
from langchain.prompts import ChatPromptTemplate  # For ChatPromptTemplate

CHORMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Answer the question based on the above context: {query_text}
"""

# Define embedding function
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # print("Query Text: ", query_text)
    
    # Prepare the DB
    Myembeddings = get_embedding_function()
    # print("Embeddings ", Myembeddings)
    
    # Load the database
    db = Chroma(persist_directory=CHORMA_PATH, embedding_function=Myembeddings)
    
    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(query_text)
    
    if len(results) == 0:
        print("Unable to find matching chunks.")
        return
    print("Found matching chunks.")
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query_text=query_text)
    print("Prompt: ", prompt)
    
    # Use HuggingFace chatbot to generate response
    # LOGIN TO HUGGINGFACE

    hf_token = os.getenv("HUGGINGFACE_TOKEN")  # Store in environment variables
    login(hf_token)
    if not hf_token:
       raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable.")
    
    # Load the model for text inference
    # Load LLaMA 2 model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    
    # Generate response
    response = text_generator(prompt, max_new_tokens=200)
    response_text = response[0]['generated_text']
        
    sources = [doc.metadata["source"] for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
if __name__ == "__main__":
    main()
