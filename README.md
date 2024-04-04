# LlamaIndex v0.10.26  
A LlamaIndex RAG sample, local LLM & embeded LLM  
for OpenAI Python API library 1.X  

# first! prepare env, install Build Tools for Visual Studio 2022  
https://visualstudio.microsoft.com/ja/downloads/  

# poetry require modules.   
poetry install --no-root, then set require modules from pyproject.toml  

# set configuration-env,  input-data(PDF), LLM model  
mkdir data/     (and set a PDF file, for RAG Search)  
git lfs clone  https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf/ --include "ELYZA-japanese-Llama-2-7b-instruct-q8_0.gguf"  
copy ELYZA-japanese-Llama-2-7b-instruct-q8_0.gguf to %Appdata%/local/llama_index/models/  

# run application command.   
python .\raglocal.py  
