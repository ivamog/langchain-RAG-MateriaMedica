# Langchain RAG MateriaMedica

This repository implements a Retrieval-Augmented Generation (RAG) system using Langchain and LlaMA 2 (Huggingface).  MD file(s) in Data directory provide context conteining some of the Homeopatic remedies description. More than one MD file can be used to augment the context.

## Setup
1. Clone the repository
2. Install dependencies
3. Configure environment variables

## Usage
If using your own MD files, run RAGforBooks.py first to re-create vector database. Then run RAGPromptOne.py passing the prompt question that can be answered by the uploaded context from MD file(s).
In this example, prompt passed to the RAGPromptOne.py can contain the question on the best homeopathic remedy for the list of symptoms. 

For example:
py RAGPromptOne.py 'What is the best remedy for the pollen allergy with stuffy nose?'

The MD file provided in Data directory enables lightwight Homeopathic repeortory as a demo and recommended remedies should not be used to cure. The quality of the answers will depend on the content of the file and model used.
