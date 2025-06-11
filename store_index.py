from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = api_key

# Load and prepare data
extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Init Pinecone client
pc = Pinecone(api_key=api_key)

# Delete old index if exists
try:
    pc.delete_index("medicalbot")
    print("Deleted existing index 'medicalbot'")
except Exception as e:
    print(f"Index delete skipped: {e}")

# Create new index
pc.create_index(
    name="medicalbot",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
print("Created new index 'medicalbot'")

# Wait for index to be ready
while True:
    description = pc.describe_index("medicalbot")
    if description.status['ready']:
        print("Index is ready.")
        break
    print("Waiting for index to be ready...")
    time.sleep(2)

# Upload to index
index_name = "medicalbot"
print(f"Storing {len(text_chunks)} chunks to Pinecone index '{index_name}'...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)
print("Documents successfully stored to Pinecone!")
