# This script uses Vector database as a context for answering the question, using LangChain and OpenAI.
# Works with RAGforBooks.py which saves vector embeddings to a Chroma database.
#
#
# # Make sure to have OpenAI_API_KEY set in the environment variables.
#
# To run the script:
# py RAGPromptOpenAI.py '<Your query text>'
# 

import argparse
import os
import tiktoken

from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
System: You are a homeopathy doctor that provides accurate and concise answers taking into account the given context.

User: {query_text}

Context: {context}

Please provide a concise and accurate answer based on the context provided.
"""

@dataclass
class Document:
    page_content: str
    metadata: dict

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text to process")
    args = parser.parse_args()

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize Chroma client
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Search for relevant documents
    results = db.similarity_search(args.query_text)
    
    # Format context from results
    context = "\n\n".join([doc.page_content for doc in results])
    print("Context:", context)	
    
	# Count tokens in context

    def count_tokens(text, model="gpt-4-turbo"):
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

   
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the prompt first
    formatted_prompt = prompt.format(
        query_text=args.query_text,
        context=context
    )
    
    prompt_token_count = count_tokens(formatted_prompt)
    print(f"Estimated tokens in prompt: {prompt_token_count}")
    
    # Initialize OpenAI chat model
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )

    # Create messages from prompt
    messages = prompt.format_messages(
        query_text=args.query_text,
        context=context
    )

    # Get response from model
    response = model.invoke(messages)
    
    # Count output tokens
    output_token_count = count_tokens(response.content)
    print(f"Estimated output tokens: {output_token_count}")

    # Print response
    print("\nQuery:", args.query_text)
    print("\nResponse:", response.content)

if __name__ == "__main__":
    main()
