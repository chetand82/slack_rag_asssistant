import os
import json
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import boto3

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # Replace with your region
)

# Use Claude v2 embeddings (or similar) through Bedrock
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"  # or another embed model if you prefer
)

# Load synthetic Slack messages from file
slack_conversation = r'D:\Chetan\Project\slack_rag_assistant\slack_rag_asssistant\scripts\synthetic_slack_threads_1000.json'
with open(slack_conversation, "r", encoding="utf-8") as f:
    slack_data = json.load(f)

'''
with open(r'D:\Chetan\Project\slack_rag_assistant\slack_rag_asssistant\scripts\test.txt', "r", encoding="utf-8") as f:
    chetan = json.load(f)
'''
# Convert each message into a document
documents = []
for thread in slack_data:
    for message in thread["messages"]:
        content = message.get("text", "")
        if content.strip():
            documents.append(Document(page_content=content))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Embed and create FAISS index
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Save the FAISS vectorstore
vectorstore.save_local("vectorstore")

print(f"✅ Embedded and saved {len(split_docs)} chunks to 'vectorstore/'")
