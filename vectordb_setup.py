__author__ = "Jon Ball"
__version__ = "December 2023"

from langchain.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import chromadb
import openai
import torch
import random
import os
_ = load_dotenv("openai.env")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # Access local chroma client
    persistent_client = chromadb.PersistentClient(path="chroma")
    # Load checklists
    checklists, metadatas = load_checklists()
    # Set up local chroma dbs
    print("Implementing vector db over checklists...")
    print('   ...setting up local chromadb collections with each of 3 SGPT models from HuggingFace...')
    # db for 125M sentence GPT model
    print("   ...125M...")
    model_name = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
    embedding = hf_embed(model_name)
    vectordb = load_vectordb("125M", embedding, checklists, metadatas)
    # db for 1.3B sentence GPT model
    print("   ...1.3B...")
    model_name = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit"
    embedding = hf_embed(model_name)
    vectordb = load_vectordb("1_3B", embedding, checklists, metadatas)
    # db for 2.7B sentence GPT model
    print("   ...2.7B...")
    model_name = "Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit"
    embedding = hf_embed(model_name)
    vectordb = load_vectordb("2_7B", embedding, checklists, metadatas)
    # db for OpenAI text-embedding-ada-002 model
    print('   ...setting up local chromadb collection with OpenAI "text-embedding-ada-002"...')
    model_name = "text-embedding-ada-002"
    embedding = openai_embed(model_name)
    vectordb = load_vectordb("openai", embedding, checklists, metadatas)
    print("Done.")


def load_vectordb(collection_name, embedding, texts, metadatas):
    vectordb = Chroma.from_texts(
        collection_name=collection_name,
        embedding=embedding,
        metadatas=metadatas,
        texts=texts,
        persist_directory="chroma")
    return vectordb


def hf_embed(model_name):
    embedding = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        )
    return embedding


def openai_embed(model_name):
    embedding = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    return embedding


def load_checklists():
    print("Loading research reporting guideline checklists...")
    checklists = []
    metadatas = []
    for dirname, dirpath, filenames in os.walk("checklists"):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            with open(os.path.join(dirname, filename), "r") as infile:
                checklist = infile.read()
            checklists.append(checklist)
            metadatas.append({"name": filename[:-4], "checklist": checklist})
    print(f"   ...loaded {len(checklists)} checklists.")
    return checklists, metadatas


if __name__ == "__main__":
    main()
