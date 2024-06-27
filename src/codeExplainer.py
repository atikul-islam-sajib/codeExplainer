import os
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


class Explainer:
    def __init__(self, url=None, model="OpenAI"):
        self.url = url
        self.model = model

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
                temperature=1.0,
                model_name="gpt-3.5-turbo",
                openai_api_key=self.access_api_key(),
            )

            return self.model

        else:
            raise Exception("Model not found".capitalize())
