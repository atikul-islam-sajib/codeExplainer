import os
import sys
import warnings
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

load_dotenv()

sys.path.append("src/")

from utils import dump, load, config


class Explainer:
    def __init__(self, url=None, model="OpenAI"):
        self.url = url
        self.model = model

        self.CONFIG = config()

    def access_api_key(self):
        try:
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            return self.OPENAI_API_KEY

        except Exception as e:
            print(e)
            raise Exception("API key not found")

    def model_init(self):
        if self.model == "OpenAI":
            self.model = OpenAI(
                temperature=self.CONFIG["OpenAI"]["temperature"],
                model_name=self.CONFIG["OpenAI"]["model_name"],
                openai_api_key=self.access_api_key(),
            )

            return self.model

        else:
            raise Exception("Model not found".capitalize())
