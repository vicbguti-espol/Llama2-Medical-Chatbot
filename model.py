from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from deep_translator import GoogleTranslator
import time


DB_FAISS_PATH = 'vectorstore/db_faiss'

# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

custom_prompt_template = """Utilice la siguiente información para responder la pregunta del usuario.
Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelva la útil respuesta a continuación y nada más.
Respuesta útil:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
    #                                    model_kwargs={'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


start_time = time.time()
# result = final_result('¿Para quién es la salud mental?')
# result = final_result('¿Cuál es la diferencia entre el conocer y el comunicarse?')
result = final_result('¿Qué es la inteligencia emocional?')
# result = final_result('¿Cómo funcionan las emociones prohibidas en los niños?')
print(result)
print("--- %s seconds ---" % (time.time() - start_time))
# chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # message = GoogleTranslator(source='auto', target='en').translate(text=raw_message) # added line
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # answer = GoogleTranslator(source='auto', target='es').translate(text=res["result"]) 
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

