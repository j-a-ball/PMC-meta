__author__ = "Jon Ball"
__version__ = "November 2023"

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import chromadb
import jinja2
import torch
import random
import json
import time
import os

def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # Access local chroma client
    persistent_client = chromadb.PersistentClient(path="chroma")
    for tup in [
        ("Muennighoff/SGPT-125M-weightedmean-nli-bitfit", "125M"), 
        ("Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit", "1_3B"), 
        ("Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit", "2_7B"),
        ("Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit", "5_8B")
        ]:
        model_name, collection_name = tup
        print(f'Creating prompts using "{model_name}" for retrieval...')
        create_prompts(persistent_client, model_name, collection_name)
        print(f'   ...prompts created for "{model_name}".')
    print("Done.")

def create_prompts(persistent_client, model_name, collection_name):
    # HF embedding model config
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding = load_embedding(model_name, model_kwargs, encode_kwargs)
    vectordb = load_vectordb(persistent_client, collection_name, embedding)
    for dirname, dirpath, filenames in os.walk("articles"):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            with open(os.path.join(dirname, filename), "r") as infile:
                article = infile.read()
            checklist = vectordb.similarity_search(article, 1)[0].metadata["checklist"]
            # generate prompt with jinja2
            output_dir = f"prompts/{collection_name}"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            jinjitsu = jinjaLoader("prompts", "checklist.prompt")
            templateVars = {"checklist": checklist, "article": article}
            prompt = jinjitsu.render(templateVars)
            # write prompt to file
            with open(f"{output_dir}/{filename[:-4]}.prompt", "w") as outfile:
                outfile.write(prompt)

def load_vectordb(persistent_client, collection_name, embedding):
    vectordb = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding,
        )
    return vectordb

def load_embedding(model_name, model_kwargs, encode_kwargs):
    embedding = SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    return embedding               


class jinjaLoader():
    # jinja2 template renderer
    def __init__(self, template_dir, template_file):
        self.templateLoader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.templateEnv = jinja2.Environment( loader=self.templateLoader )
        self.template = self.templateEnv.get_template( template_file )

    def render(self, templateVars):
        return self.template.render( templateVars )
    

if __name__ == "__main__":
    main()