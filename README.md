# PMC-meta
Using LLMs to automate research quality assessments in medical literature

To reproduce the results:

1. Run `vectordb_setup.py` to set up persistent Chroma vector databases and embedding functions for lookup. (Note: You must provide own OPENAI_API_KEY as an environ variable e.g. in `openai.env` to reproduce some embeddings.)
2. Run `create_prompts.py`, which combines articles and retrieved checklists into prompts using a jinja2 template.
3. Run `run_inference.py`, which makes 22 calls to "gpt-4-1106-preview" (128k context): 11 for prompts with HuggingFace-retrieved checklists applied to articles, and 11 for OpenAI-retrieved checklists. 
