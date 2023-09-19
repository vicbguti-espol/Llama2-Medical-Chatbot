from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from deep_translator import GoogleTranslator

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader,)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=50)
    # print('Getting docs...')
    docs = text_splitter.split_documents(documents)
    print(docs[0])
    # texts = [x.page_content for x in docs]
    # print('Getting translations...')
    # ts_texts = [GoogleTranslator(source='auto', target='en').translate(text=x) for x in texts]

    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    #                                    model_kwargs={'device': 'cpu'})
    # embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large',
    #                                    model_kwargs={'device': 'cpu'})
    # embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small',
    #                                    model_kwargs={'device': 'cpu'})


    # db = FAISS.from_documents(docs, embeddings)
    # # db = FAISS.from_texts(ts_texts, embeddings)
    # db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

