import os
import sys
import argparse
import warnings
import traceback
import chainlit as cl
from dotenv import load_dotenv
from git import Repo
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

load_dotenv()

sys.path.append("src/")

from utils import dump, load, config, clean_folder, CustomException
from template import template


class Explainer:
    def __init__(self, url=None, model="OpenAI"):
        self.url = url
        self.model = model

        self.CONFIG = config()
        self.clean_folder = clean_folder()

        os.makedirs(self.CONFIG["path"]["CODE_PATH"], exist_ok=True)

    def access_api_key(self):
        try:
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

            return {
                "OPENAI_API_KEY": self.OPENAI_API_KEY,
                "HUGGINGFACE_API_TOKEN": self.API_TOKEN,
            }

        except Exception as e:
            print(e)
            raise Exception("API key not found")

    def model_init(self):
        if self.model == "OpenAI":
            self.model = OpenAI(
                temperature=self.CONFIG["OpenAI"]["temperature"],
                model_name=self.CONFIG["OpenAI"]["model_name"],
                openai_api_key=self.access_api_key()["OPENAI_API_KEY"],
            )

            return self.model

        else:
            raise Exception("Model not found".capitalize())

    def download_source_code(self):
        if os.path.exists(self.CONFIG["path"]["CODE_PATH"]):
            self.CODE_PATH = self.CONFIG["path"]["CODE_PATH"]
            self.URL = self.CONFIG["sourcecode"]["url"]

            try:
                if self.URL:
                    Repo.clone_from(url=self.URL, to_path=self.CODE_PATH)
                else:
                    raise CustomException("URL not found".capitalize())

            except CustomException as exception:
                print("The exaceptio is", exception)
                traceback.print_exc()
        else:
            print("Try it again to access further functionalities".capitalize())

    def generate_tokens(self):
        self.extension = self.CONFIG["analysis"]["filenames"]

        try:

            if (
                self.extension == "py"
                or self.extension == "java"
                or self.extension == "cpp"
            ):
                self.loader = DirectoryLoader(
                    path=os.path.join(
                        self.CONFIG["path"]["CODE_PATH"],
                    ),
                    glob="**/*.{}".format(self.CONFIG["analysis"]["filenames"]),
                )

                self.documents = self.loader.load()

                return self.documents

            else:
                raise CustomException(
                    "File extension not found, check the config yaml file".capitalize()
                )

        except CustomException as exception:
            print("The exaceptio is", exception)
            traceback.print_exc()

    def database_init(self, documents=None):
        self.database = self.CONFIG["vectorstores"]

        if isinstance(documents, list):
            if self.database["Chroma"]:
                os.makedirs("./DB", exist_ok=True)

                self.vectordb = Chroma.from_documents(
                    documents=documents,
                    persist_directory="./DB",
                    embedding=(
                        OpenAIEmbeddings()
                        if self.CONFIG["embeddings"]["OpenAI"]
                        else HuggingFaceBgeEmbeddings(
                            model_name=self.CONFIG["embeddings"]["HuggingFace"]
                        )
                    ),
                )

                self.vectordb = Chroma(
                    persist_directory="./DB",
                    embedding_function=(
                        OpenAIEmbeddings()
                        if self.CONFIG["embeddings"]["OpenAI"]
                        else HuggingFaceBgeEmbeddings(
                            model_name=self.CONFIG["embeddings"]["HuggingFace"]
                        )
                    ),
                )

                print("Chroma is done".capitalize())

                return self.vectordb

            else:
                self.vectordb = FAISS.from_documents(
                    documents=documents,
                    embedding=(
                        OpenAIEmbeddings()
                        if self.CONFIG["embeddings"]["OpenAI"]
                        else HuggingFaceBgeEmbeddings(
                            model_name=self.CONFIG["embeddings"]["HuggingFace"]
                        )
                    ),
                )

                print("FAISS is done".capitalize())

                return self.vectordb

        else:
            raise ValueError("documents must be a list".capitalize())

    def generate_embeddings(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CONFIG["chunks"]["chunk_size"],
            chunk_overlap=self.CONFIG["chunks"]["chunk_overlap"],
        )

        self.documents = self.text_splitter.split_documents(
            documents=self.generate_tokens()
        )

        dump(
            value=self.documents,
            filename=os.path.join(self.CONFIG["path"]["DATA_PATH"], "documents.pkl"),
        )

        print(
            "chunking is done and stored in the folder of {}".format(
                self.CONFIG["path"]["DATA_PATH"]
            )
        )

        return self.documents

    def define_prompt_and_memeory(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "history"], template=template
        )
        self.memory = ConversationBufferMemory(
            input_key="question", memory_key="history"
        )

        return {"prompt": self.prompt, "memory": self.memory}

    def chatExplainer(self):
        try:
            self.vectordb = self.database_init(documents=self.generate_embeddings())
        except ValueError as exception:
            print("The exaceptio is", exception)
            traceback.print_exc()
        except Exception as exception:
            print("The exaceptio is", exception)
            traceback.print_exc()

        self.retriever = self.vectordb.as_retriever()

        self.chain = RetrievalQA.from_chain_type(
            llm=self.model_init(),
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.define_prompt_and_memeory()["prompt"],
                "memory": self.define_prompt_and_memeory()["memory"],
            },
        )

        self.chat_limit = self.CONFIG["chatExplainer"]["chat_limit"]

        while self.chat_limit > 0:
            self.query = input("Query: ")
            print("Answer: ", self.chain(self.query)["result"])

            self.chat_limit -= 1

        print("The chat limit is completed. Try again !".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain the code".capitalize())
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yml",
        help="Define the config file".capitalize(),
    )
    parser.add_argument("--chat", action="store_true", help="Chat model".capitalize())

    args = parser.parse_args()

    if args.config and args.chat:

        explainer = Explainer()

        explainer.download_source_code()
        explainer.generate_tokens()

        explainer.chatExplainer()

    else:
        print("The config file is not defined".capitalize())
