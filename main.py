import os
import sys

import gradio
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []


def call_interface(query):
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return result['answer']


def run_cli_mode():
    while True:
        query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        print(call_interface(query))


def run_gui_mode():
    demo = gradio.Interface(fn=call_interface, inputs="text", outputs="text", title="My Custom ChatGPT")
    demo.launch(share=True)


if __name__ == "__main__":
    # run_cli_mode()
    run_gui_mode()
