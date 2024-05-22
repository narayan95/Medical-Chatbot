from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
DB_FAISS_PATH = 'vectorstore/db_faiss'
# llama 2-> 4096 token limit . 16000 english characters
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 500,
        temperature = 0.5
    )
    return llm

#Retrieval QA Chain creating a chain to answer
def retrieval_qa_chain(llm, prompt, db):
    #question_answer_chain = create_stuff_documents_chain(llm, prompt)
    #qa_chain = create_retrieval_chain(db.as_retriever, question_answer_chain)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True, #llama2 knowledge ko ignore kro
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    #response= qa_result.invoke({"input": query})
    response = qa_result({'query': query})
    return response

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"LLMMathChain": "Albert Einstein", "Chatbot": "Virtual Doctor"}
    return rename_dict.get(orig_author, orig_author)

# Or use a local image file located in your public directory

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting your session...")
    await msg.send()
    msg.content = "Hi, I am the Vitual Doctor. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    #res = await chain.ainvoke(message.content, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: {str(sources)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

