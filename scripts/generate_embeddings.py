import os
import json
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import boto3

# === Configuration ===
VECTORSTORE_DIR = "vectorstore"
SLACK_FILE_PATH = r"D:\Chetan\Project\slack_rag_assistant\slack_rag_asssistant\scripts\synthetic_slack_threads_1000.json"
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v1"

# === Step 1: Initialize Bedrock client ===
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=BEDROCK_REGION,
)

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id=BEDROCK_MODEL_ID,
)

# === Step 2: Load synthetic Slack messages ===
with open(SLACK_FILE_PATH, "r", encoding="utf-8") as f:
    slack_data = json.load(f)

documents = []

# === Step 3: Parse Slack messages from known structures ===
if isinstance(slack_data, dict):
    # Try nested structure: { "threads": [ { "messages": [ {"text": "..."} ] }, ... ] }
    if "threads" in slack_data and isinstance(slack_data["threads"], list):
        for thread in slack_data["threads"]:
            for message in thread.get("messages", []):
                content = message.get("text", "")
                if content.strip():
                    documents.append(Document(page_content=content))
    # Try flat: { "messages": [ {"text": "..."} ] }
    elif "messages" in slack_data and isinstance(slack_data["messages"], list):
        for message in slack_data["messages"]:
            content = message.get("text", "")
            if content.strip():
                documents.append(Document(page_content=content))
    else:
        raise ValueError("Unsupported dictionary structure in Slack JSON.")
elif isinstance(slack_data, list):
    # Fallback if top-level is a list
    for item in slack_data:
        if isinstance(item, dict) and "messages" in item:
            for message in item["messages"]:
                content = message.get("text", "")
                if content.strip():
                    documents.append(Document(page_content=content))
        elif isinstance(item, dict) and "text" in item:
            content = item.get("text", "")
            if content.strip():
                documents.append(Document(page_content=content))
else:
    raise ValueError("Unsupported Slack data format.")

# === Step 4: Split into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# === Step 5: Embed and create FAISS vectorstore ===
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local(VECTORSTORE_DIR)

print(f"✅ Embedded and saved {len(split_docs)} chunks to '{VECTORSTORE_DIR}/'")
