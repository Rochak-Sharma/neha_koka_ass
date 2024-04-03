import os
from typing import Any, Dict, List
from langchain import OpenAI, LLMChain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import (
    PlayWrightBrowserToolkit,
    PlaywrightBrowserToolkitRun,
)
from langchain.tools import Tool
from langchain.utilities.youtube import SearchYouTube
from langchain.vectorstores import STAPLEVECTORS
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

os.environ["OPENAI_API_KEY"] = "sk-abcd1234567890efghijklmnopqrstuvwxyz"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBcDeFGHIJKLMNOPQRSTUVWXYZ0123456"
os.environ["GOOGLE_CSE_ID"] = "012345678901234567890:abcdefghij"

search = SearchYouTube(
    engine="google",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    google_cse_id=os.environ["GOOGLE_CSE_ID"],
)
search_tool = Tool(
    name="Search YouTube",
    description="Search for relevant YouTube videos",
    func=search.run,
)

llm = OpenAI(temperature=0.4)

clip_model = HuggingFaceHub(repo_id="openai/clip-vit-base-patch32", model_kwargs={"device": "cpu"})

toolkit = PlayWrightBrowserToolkit(llm=llm, clip_model=clip_model)
toolkit_run = PlaywrightBrowserToolkitRun(toolkit=toolkit)

memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    toolkit=toolkit,
    toolkit_run=toolkit_run,
    memory=memory,
    tools=[search_tool],
    llm=llm,
)

def chat(input_text_or_image: str) -> Dict[str, Any]:
    response = agent(input_text_or_image)
    return response

print(chat("How do I solve a quadratic equation?"))
print(chat("path/to/image.jpg"))
