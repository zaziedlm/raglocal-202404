import logging
import os
import sys

from llama_index.core import (
    # LLMPredictor, # The LLMPredictor object is no longer intended to be used by users.
    PromptTemplate,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# Read document from data-folder.
document = SimpleDirectoryReader(os.path.join(os.path.dirname(__file__), "data")).load_data()

from llama_index.llms.llama_cpp import LlamaCPP

#model_path = f"models/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/ELYZA-japanese-Llama-2-7b-instruct-q8_0.gguf"
model_path = f"ELYZA-japanese-Llama-2-7b-instruct-q8_0.gguf"
llm = LlamaCPP(
    model_url=model_path,
    temperature=0.1,
    model_kwargs={"n_ctx": 4096, "n_gpu_layer": 32},
)
EMBEDDING_DEVICE = "cpu"

# embed_model_name = "sentence-transformers/all-mpnet-base-v2"
embed_model_name = "intfloat/multilingual-e5-large"
cache_folder = "./sentence_transformers"

embed_model = HuggingFaceEmbedding(
    model_name=embed_model_name,
    cache_folder=cache_folder,
    device=EMBEDDING_DEVICE,
)

# ServiceContext was deprecated, globally in the Settings
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
Settings.num_output = 256
Settings.context_window = 3900

index = VectorStoreIndex.from_documents(
    document,
)

# 質問
temp = """
[INST]
<<SYS>>
以下の「コンテキスト情報」を元に「質問」に回答してください。
なお、コンテキスト情報に無い情報は回答に含めないでください。
また、コンテキスト情報から回答が導けない場合は「分かりません」と回答してください。
<</SYS>>
# コンテキスト情報
---------------------
{context_str}
---------------------

# 質問
{query_str}

[/INST]
"""

query_engine = index.as_query_engine(
    similarity_top_k=5,
    text_qa_template=PromptTemplate(temp),
)

while True:
    req_msg = input("\n## Question: ")
    if req_msg == "":
        continue
    res_msg = query_engine.query(req_msg)
    res_msg.source_nodes[0].text
    print("\n## Answer: \n", str(res_msg).strip())
