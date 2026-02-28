from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class LocalRAG:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.vector_db = None
        # Using a lightweight embedding model that can be routed to the NPU
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def ingest_documents(self):
        """Loads PDFs and chunks them for local vector storage."""
        print("Reading local enterprise documents...")
        loader = PyPDFDirectoryLoader(self.data_dir)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        print("Embedding documents locally...")
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory="../models/chroma_db"
        )
        print("Offline Indexing Complete.")

    def retrieve_context(self, query, top_k=3):
        """Retrieves the most relevant local document chunks."""
        if not self.vector_db:
            self.vector_db = Chroma(persist_directory="../models/chroma_db", embedding_function=self.embeddings)
            
        results = self.vector_db.similarity_search(query, k=top_k)
        context = "\n".join([doc.page_content for doc in results])
        return context
