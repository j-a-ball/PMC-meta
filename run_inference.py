__author__ = "Jon Ball"
__version__ = "November 2023"

from openai_utils import (start_chat, user_turn, system_turn)
from tqdm import tqdm
import torch
import random
import time
import os

def main():
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    # loop over prompts
    if not os.path.exists("completions"):
        os.mkdir("completions")
    print("Generating completions...")
    for dirname, dirpath, filenames in os.walk("prompts/2_7B"):
        for filename in tqdm([f for f in filenames if f.endswith(".prompt")]):
            with open(os.path.join(dirname, filename), "r") as infile:
                prompt = infile.read()
            chat = start_chat("You are an expert biostastician, methodologist, and reviewer of medical articles. You are reviewing a biomedical or health research article, according to a research reporting guideline checklist. RETURN A COMPLETED VERSION OF THE CHECKLIST.")
            chat = user_turn(chat, prompt)
            try:
                chat = system_turn(chat, model="gpt-4-1106-preview")
            except TimeoutError:
                # try again
                time.sleep(random.randint(1, 3))
                chat = system_turn(chat, model="gpt-4-1106-preview")
            with open(f"completions/{filename[:-7]}.txt", "w") as outfile:
                outfile.write(chat[-1]["content"])
    print("Completions saved.")

if __name__ == "__main__":
    main()