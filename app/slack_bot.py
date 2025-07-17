# Slack message handler
import os
import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.llms import Bedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM


# Environment variables
# SLACK_BOT_TOKEN
# slack app configurations token
#

#SLACK_APP_TOKEN - explicitly generate while creating new app. Yet to figure out how to do for existing APP
#

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Setup Bedrock client
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")  # change region if needed

# Claude model on Bedrock
def get_claude_llm():
    return BedrockLLM(
        client=bedrock_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={
            "max_tokens": 1000,
            "temperature": 0.3,
        }
    )

# Load vector store (FAISS)
def get_qa_chain():
    embeddings = BedrockEmbeddings(client=bedrock_client)
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    llm = get_claude_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

qa_chain = get_qa_chain()

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

@app.event("app_mention")
def handle_app_mention_events(body, say):
    user = body["event"]["user"]
    thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
    text = body["event"]["text"]

    # Remove bot mention
    query = " ".join(word for word in text.split() if not word.startswith("<@"))

    say(f"Hi <@{user}>, checking that for you...", thread_ts=thread_ts)

    try:
        result = qa_chain.run(query)
        say(result, thread_ts=thread_ts)
    except Exception as e:
        say(f"❗ Error: {e}", thread_ts=thread_ts)

def start():
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
    start()
