# Medical Methodologies: Retrieving research reporting guidelines to augment LLM-generated feedback on medical and health research manuscripts

Current-gen large language models (LLMs) like GPT-4 (OpenAI 2023) have been shown to be capable of providing feedback on researchers’ manuscripts (Liang et al. 2023). However, prior implementations of LLM feedback systems have truncated the full text of manuscripts to fit context constraints, while also prompting LLMs with acontextual feedback schemes. Instead, this study proposes using LLMs with wider context windows to learn domain-specific evaluation schemes in-context and then review the full text of manuscripts. In medical and health research, for example, there are numerous peer-reviewed best practice checklists covering different study types. These best practice documents can be encoded into vector databases, retrieved, and finally incorporated into LLM prompts, in order to generate grounded and contextually relevant feedback on researchers’ manuscripts.

To reproduce the results:

1. Run `vectordb_setup.py` to set up persistent Chroma vector databases and embedding functions for lookup. (Note: You must provide own OPENAI_API_KEY as an environ variable e.g. in `openai.env` to reproduce some embeddings.)
2. Run `create_prompts.py`, which combines articles and retrieved checklists into prompts using a jinja2 template.
3. Run `run_inference.py`, which makes 22 calls to "gpt-4-1106-preview" (128k context): 11 for prompts with HuggingFace-retrieved checklists applied to articles, and 11 for OpenAI-retrieved checklists. 
