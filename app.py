import sys
import traceback
from langchain.chains import RetrievalQA
import chainlit as cl

sys.path.append("src/")
from src.codeExplainer import Explainer

explainer = Explainer()
explainer.download_source_code()
explainer.generate_tokens()


@cl.on_chat_start
async def start():
    try:
        vectordb = explainer.database_init(documents=explainer.generate_embeddings())
    except ValueError as exception:
        print("The exception is", exception)
        traceback.print_exc()
    except Exception as exception:
        print("The exception is", exception)
        traceback.print_exc()

    retriever = vectordb.as_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=explainer.model_init(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": explainer.define_prompt_and_memeory()["prompt"],
            "memory": explainer.define_prompt_and_memeory()["memory"],
        },
    )

    cl.user_session.set("llm_chain", chain)


@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    response = await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(response["result"]).send()
