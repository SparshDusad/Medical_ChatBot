from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone v3
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-bot"
dimension = 384

# Create index if doesn't exist
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Load docs and embeddings
docs = load_pdf_file("data/")
filtered_docs = filter_to_minimal_docs(docs)
chunks = text_split(filtered_docs)
embeddings = download_hugging_face_embeddings()

# Create vector store and upload
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace="",  # optional
)
