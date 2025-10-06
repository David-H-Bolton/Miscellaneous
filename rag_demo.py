from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("my_docs.txt")
docs = loader.load()

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Create embeddings (local model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

# 5. Use Ollama for local generation
llm = Ollama(model="mistral")

# 6. Build RAG pipeline
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 7. Ask questions
while True:
    query = input("\nAsk me something (or type 'exit'): ")
    if query.lower() == "exit":
        break
    print("\nAnswer:", qa.run(query))