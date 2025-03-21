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
from openai import OpenAI

from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
System: You are a homeopathy doctor that provides accurate and concise answers taking into account the given context. You must not claim that any of the remedies can cure terminal deseises.

User: {query_text}

Context: {context}

Please provide a concise and accurate answer based on the context provided.
"""

BLACKLISTED_WORDS = ["violence", "self-harm"]

# Add these constants for guardrails
MAX_TOTAL_TOKENS = 4096  # for gpt-3.5-turbo
MAX_INPUT_LENGTH = 500  # characters
MIN_CONTEXT_LENGTH = 50  # characters
MAX_RETRIES = 2

@dataclass
class Document:
    page_content: str
    metadata: dict

def validate_input(query_text: str) -> bool:
    """Validate the input query"""
    if not query_text or len(query_text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Query must be between 1 and {MAX_INPUT_LENGTH} characters")
    return True

def validate_context(context: str) -> bool:
    """Validate the context length"""
    if len(context) < MIN_CONTEXT_LENGTH:
        raise ValueError("Insufficient context found")
    return True

def is_safe_text(text):
    """Checks if the text is safe using OpenAI's moderation API."""
    try:
        client = OpenAI()  # This will use your OPENAI_API_KEY environment variable
        response = client.moderations.create(input=text)
        return not response.results[0].flagged  # Flagged True if inappropriate
    except Exception as e:
        print(f"Moderation API error: {e}")
        return False  # Assume unsafe if API fails

def contains_blacklisted_words(text):
    # Checks if the text contains blacklisted words."""
    lower_text = text.lower()
    return any(word in lower_text for word in BLACKLISTED_WORDS)

def main():
    try:
        # Initialize argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("query_text", type=str, help="The query text to process")
        args = parser.parse_args()

        # Validate input length
        validate_input(args.query_text)

        # Check if the query text is safe
        if not is_safe_text(args.query_text):
            print("Error: The query contains inappropriate content.")
            return
        if contains_blacklisted_words(args.query_text):
            print("Error: The query contains restricted topics.")
            return
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Chroma client with error handling
        try:
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector database: {str(e)}")

        # Search for relevant documents with retry
        for attempt in range(MAX_RETRIES):
            try:
                results = db.similarity_search(args.query_text)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"Failed to search database after {MAX_RETRIES} attempts")
                continue

        # Format and validate context
        context = "\n\n".join([doc.page_content for doc in results])
        validate_context(context)
        print("Context:", context)

        # Count tokens and check limits
        def count_tokens(text, model="gpt-3.5-turbo"):
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))

        # Create and format prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        formatted_prompt = prompt.format(
            query_text=args.query_text,
            context=context
        )

        # Check token limits
        prompt_token_count = count_tokens(formatted_prompt)
        if prompt_token_count >= MAX_TOTAL_TOKENS:
            raise ValueError(f"Input too long: {prompt_token_count} tokens exceeds limit of {MAX_TOTAL_TOKENS}")
        print(f"Estimated tokens in prompt: {prompt_token_count}")
        
        # Initialize OpenAI chat model with error handling
        try:
            model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                request_timeout=30  # Add timeout
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")

        # Create messages and get response with retry
        messages = prompt.format_messages(
            query_text=args.query_text,
            context=context
        )

        # Get response with retry
        for attempt in range(MAX_RETRIES):
            try:
                response = model.invoke(messages)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"Failed to get response after {MAX_RETRIES} attempts")
                continue

        # Validate response
        if not response or not response.content:
            raise ValueError("Empty response received from model")

        # Check tha response is safe
        if not is_safe_text(response.content):
            print("Warning: AI response contains inappropriate content.")
            return
        
        # Count and validate output tokens
        output_token_count = count_tokens(response.content)
        print(f"Estimated output tokens: {output_token_count}")

        if output_token_count + prompt_token_count > MAX_TOTAL_TOKENS:
            raise ValueError("Total token count exceeds limit")

        # Print results
        print("\nQuery:", args.query_text)
        print("\nResponse:", response.content)

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
