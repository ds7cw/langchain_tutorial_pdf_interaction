import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_ollama.embeddings import OllamaEmbeddings # Local LLM
# from langchain_community.llms import Ollama # Local LLM
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings() # embeddings = OllamaEmbeddings(model='llama2') # Local LLM
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever
    # The lower the temperature, the more conservative the answer
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o') # llm = Ollama(model='llama2')

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain


if __name__ == '__main__':
    qa_chain = setup_qa_system('my_pdf_document.pdf')

    while True:
        question = input('\nAsk a question (Q to exit): ')
        if question.lower == 'q':
            break

        answer = qa_chain.invoke(question)
        print('Answer: ')
        print(answer)
