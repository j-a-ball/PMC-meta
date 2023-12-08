__author__ = "Jon Ball"
__version__ = "November 2023"

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import torch
import random
import os


def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # Load checklists
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
    print("Implementing vector db over checklists...")
    # Set up local chroma dbs
    print("   ...setting up local chromadbs with each of 4 SGPT model sizes...")
    if not os.path.exists("chroma"):
        os.mkdir("chroma")
    setup_chromas(checklists, metadatas)
    print("Done.")

def setup_chromas(texts, metadatas):
    # db for 125M sentence GPT model
    print("   ...125M...")
    model_name = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
    vectordb = make_chromadb(model_name, texts, "125M", metadatas)
    # db for 1.3B sentence GPT model
    print("   ...1.3B...")
    model_name = "Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit"
    vectordb = make_chromadb(model_name, texts, "1_3B", metadatas)
    # db for 2.7B sentence GPT model
    print("   ...2.7B...")
    model_name = "Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit"
    vectordb = make_chromadb(model_name, texts, "2_7B", metadatas)
    # db for 5.8B sentence GPT model
    print("   ...5.8B...")
    model_name = "Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit"
    vectordb = make_chromadb(model_name, texts, "5_8B", metadatas)
    
def make_chromadb(model_name, texts, collection_name, metadatas):
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    vectordb = Chroma.from_texts(
        texts=texts,
        collection_name=collection_name,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory="chroma")
    return vectordb


if __name__ == "__main__":
    main()
