import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment or .env file")

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",
    temperature=0.7
)

response = chat([HumanMessage(content="Hi, test GPT-4")])
print(response.content)
