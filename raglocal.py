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

