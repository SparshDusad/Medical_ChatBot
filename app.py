from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# Internal modules
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# LangChain + Pinecone + Google GenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Init
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "medical-bot"

# Set up Pinecone SDK (v3)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Get embeddings
embeddings = download_hugging_face_embeddings()

# ✅ New SDK-compatible PineconeVectorStore
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",  # Or "page_content" if that's what your chunks use
    namespace="",     # Optional
)

# RAG Chain Setup
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", max_tokens=500)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
qa_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# Flask Endpoints
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    response = rag_chain.invoke({"input": msg})
    return str(response.get("answer", "No response"))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
