from langchain.chat_models import init_chat_model
from typing import Union
from fastapi import FastAPI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel
from IPython.display import Image, display
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import getpass
import os
import dotenv
import json
import uuid
import asyncio
import sys
import threading


root_path = "~"
env_path = f"{os.path.expanduser(root_path)}/fold/.env"
dotenv.load_dotenv(env_path)
mcp_client: Union[MultiServerMCPClient, None] = None
mcp_tools: Union[list, None] = None

tool = TavilySearch(max_results=2)
tools = [tool]


val = dotenv.get_key(
    dotenv_path=env_path,
    key_to_get="GROQ_API_KEY",
)
if not os.environ.get("GROQ_API_KEY") and val:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Provide your Groq model Password :")
    sys.exit(2)
else:
    print("loaded env variable")
    llm = init_chat_model(model="llama3-8b-8192", model_provider="groq", api_key=val)
    llm_with_tools = llm.bind_tools(tools)


memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(State)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mcp_server()
    global App
    App = graph.compile(checkpointer=memory)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=["*"],
)


def sequential_thinking(state: State):
    tools = mcp_client.get_tools()
    return {"messages": llm.invoke(state["messages"])}


def tools_cond(state: State):
    if isinstance(state, list):
        lst_msg = state[-1]
    elif messages := state.get("messages", []):
        lst_msg = messages[-1]
    else:
        raise ValueError("Bad Graph State")
    if len(lst_msg.tool_calls) == 0:
        return "END"
    else:
        return "tools"
    # lst_msg["tool_calls"]


def convert_to_markdown_node(state: State):
    msg = state["messages"]

    template_prompt = """
    Convert the following text to properly formatted Markdown. Do not include any explanations or introductory phrases. Output only the converted Markdown:
    Text: {text}
    """

    template = ChatPromptTemplate.from_template(template_prompt)

    seq_llm_with_tools = llm.bind_tools(tools)
    chain = template | seq_llm_with_tools

    print(msg[1].content)
    output = chain.invoke({"text": msg[-1].content})
    return {"messages": output}


async def connect_to_mcp_server():
    global mcp_client, mcp_tools, App
    try:
        async with MultiServerMCPClient(
            {
                "sequential-thinking": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
                }
            }
        ) as client:
            if client:
                mcp_client = client
                mcp_tools = client.get_tools()
                graph.add_node("seq_thought", sequential_thinking)
                graph.add_node("markdown", convert_to_markdown_node)
                graph.add_node("tools", ToolNode(mcp_tools))
                graph.add_edge(START, "seq_thought")
                graph.add_edge("seq_thought", "markdown")

                graph.add_conditional_edges(
                    "markdown", tools_cond, {"END": END, "tools": "tools"}
                )
                graph.add_edge("tools", "seq_thought")
            else:
                print("mcp client no initialize, but moving on anyways")
    except KeyboardInterrupt as ke:
        print("Keyboard Interrupt", ke)


class Chat(BaseModel):
    text: str
    search: Union[bool, None] = None


class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools = {tool.name: tool for tool in tools}
        print("here", self.tools)

    def get_tools(self):
        return self.tools

    def __call__(self, input):
        if messages := input.get("messages", []):
            last_msg = messages[-1]
        else:
            raise ValueError("message not provided")

        outputs = []
        for tool_call in last_msg.tool_calls:
            tool_result = self.tools[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}


tool_node = BasicToolNode(tools)


class ChatBot:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state):
        return {"messages": [self.llm.invoke(state["messages"])]}


def route_tool(state: State):
    if isinstance(state, list) and len(state) > 0:
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError("""state -> messages is not provided or null""")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# graph.add_node("tools", tool_node)

# chatbot = ChatBot(llm_with_tools)
# graph.add_node("chatbot", chatbot)


# graph.add_conditional_edges(
#     "chatbot",
#     route_tool,
#     # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
#     # It defaults to the identity function, but if you
#     # want to use a node named something else apart from "tools",
#     # You can update the value of the dictionary to something else
#     # e.g., "tools": "my_tools"
#     {"tools": "tools", END: END},
# )
# Any time a tool is called, we return to the chatbot to decide the next step
# graph.add_edge("tools", "chatbot")


@app.post("/talk")
async def talk(chat: Chat):

    value = ""
    msg = [HumanMessage(chat.text)]

    # human_command = Command(resume={"data": human_response})

    # def streaming_response():
    #     for event in App.stream(
    #         {"messages": chat.text},
    #         config={"configurable": {"thread_id": "1234"}},
    #         stream_mode="values",
    #     ):
    #         if "messages" in event:
    #             print("here", event["messages"][-1].content)
    #             yield event["messages"][-1].content

    res = await App.ainvoke(
        {"messages": msg}, config={"configurable": {"thread_id": "1234"}}
    )
    return res["messages"][-1].content


print("running")

if __name__ == "__main__":
    subprocess.run(["fastapi", "dev", "main.py"])
    p = subprocess.Popen(["fastapi", "dev", "main.py"])
    p.communicate()
