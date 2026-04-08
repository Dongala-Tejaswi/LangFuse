import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 2. Imports (Updated)
# ==============================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from groq import Groq
from langfuse import Langfuse

# ==============================
# 3. Initialize APIs
# ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# ✅ Proper Langfuse initialization
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# ==============================
# 4. Load Data
# ==============================
file_path = "data.txt"  # keep file in same folder

try:
    with open(r"C:\Users\tejas\OneDrive\Documents\Langfuse_rag+vectordb\data.txt", "r", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    raise FileNotFoundError("❌ data.txt file not found!")

documents = [Document(page_content=text)]

# ==============================
# 5. Split into Chunks
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# ==============================
# 6. Embeddings (UPDATED)
# ==============================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ==============================
# 7. Vector Store
# ==============================
vectorstore = FAISS.from_documents(docs, embedding)

# ==============================
# 8. User Query
# ==============================
query = input("\n🔍 Enter your question: ")

# ==============================
# 9. Retrieval
# ==============================
retrieved_docs = vectorstore.similarity_search(query, k=3)
context = "\n".join([doc.page_content for doc in retrieved_docs])

# ==============================
# 10. Prompt
# ==============================
prompt = f"""
You are an AI assistant. Answer ONLY from the given context.
If answer is not in context, say "I don't know".

Context:
{context}

Question:
{query}
"""

# ==============================
# 11. Call Groq LLM
# ==============================
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

answer = response.choices[0].message.content

# ==============================
# 12. Langfuse Logging (SAFE VERSION)
# ==============================
try:
    langfuse.event(
        name="RAG Query",
        input={"query": query, "context": context},
        output={"answer": answer}
    )
except Exception as e:
    print("⚠️ Langfuse logging failed:", e)


print("\n📚 Retrieved Context:\n", context)
print("\n🤖 Answer:\n", answer)