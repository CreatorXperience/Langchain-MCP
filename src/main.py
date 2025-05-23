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
from langchain_core.messages import ToolMessage, trim_messages
from langchain_tavily import TavilySearch
from pydantic import BaseModel
from IPython.display import Image, display

from langgraph.types import Command, interrupt
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages.utils import count_tokens_approximately

from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.tools import tool as lang_tool
from email.message import EmailMessage
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

import subprocess
import getpass
import os
import dotenv
import json
import uuid
import asyncio
import sys
import pprint
import base64


root_path = "~"
env_path = f"{os.path.expanduser(root_path)}/fold/.env"
dotenv.load_dotenv(env_path)
mcp_client: Union[MultiServerMCPClient, None] = None
mcp_tools: Union[list, None] = None
llm_with_tools = None
tool = TavilySearch(max_results=4)
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


memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(State)


@lang_tool
def read_gmail_messages(token: str):
    """Fetches and reads email messages from a user's Gmail inbox using
    the Gmail API. Returns message metadata and content for further processing"""
    if not token:
        response = interrupt("a gmail token is needed to read your gmail messages")
        if not response["token"]:
            raise ValueError("token not provided")
        else:
            token = response["token"]

    service = build("gmail", "v1", credentials=Credentials(token=token))
    # Important comment below
    # pylint: disable=maybe-no-member
    # pylint: disable:R1710
    messages = (
        service.users()
        .messages()
        .list(userId="allyearmustobey@gmail.com", maxResults=5)
        .execute()
    )

    msg_id = messages["messages"][0]["id"]
    msg = (
        service.users()
        .messages()
        .get(userId="allyearmustobey@gmail.com", id=msg_id, format="full")
        .execute()
    )
    return get_message_body(msg)


# print(read_gmail_messages.args_schema.model_json_schema())


# @lang_tool
# def write_gmail_message():
#     pass


# @lang_tool
# def draft_gmail_message():
#     pass


# @lang_tool
# def delete_gmail_message():
#     pass


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
    t_msg = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=500,
        start_on="human",
        allow_partial=True,
        token_counter=len,
    )
    if len(state["messages"]) > 2:
        return {"messages": llm_with_tools.invoke(t_msg)}
    return {"messages": llm_with_tools.invoke(state["messages"])}


def tools_cond(state: State):
    if isinstance(state, list):
        lst_msg = state[-1]
    elif messages := state.get("messages", []):
        lst_msg = messages[-1]
    else:
        raise ValueError("Bad Graph State")
    if len(lst_msg.tool_calls) == 0:
        return "markdown"
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

    chain = template | llm_with_tools

    # print(msg[1].content)
    output = chain.invoke({"text": msg[-1].content})
    return {"messages": output}


class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools = {tool.name: tool for tool in tools}
        # print("here", self.tools)

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
            # if tool_call["name"] == "read_gmail_messages":
            # print("here again", tool_call.args_schema.schema(), tool_call["id"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}


# tool_node = BasicToolNode(tools)


async def connect_to_mcp_server():
    global mcp_client, mcp_tools, App, llm_with_tools
    try:
        async with MultiServerMCPClient(
            {
                "sequential-thinking": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
                },
                "mcp-installer": {
                    "command": "npx",
                    "args": ["@anaisbetts/mcp-installer"],
                },
            }
        ) as client:
            if client:
                mcp_client = client
                mcp_tools = [
                    *client.get_tools(),
                    *tools,
                    TavilySearch(max_results=4),
                    read_gmail_messages,
                ]
                for tool in mcp_tools:
                    print(tool.name)
                llm_with_tools = llm.bind_tools(mcp_tools)
                graph.add_node("seq_thought", sequential_thinking)
                graph.add_node("markdown", convert_to_markdown_node)
                graph.add_node("tools", BasicToolNode(mcp_tools))
                graph.add_edge(START, "seq_thought")

                graph.add_conditional_edges(
                    "seq_thought",
                    tools_cond,
                    {"markdown": "markdown", "tools": "tools"},
                )
                graph.add_edge("tools", "seq_thought")
                graph.add_edge("markdown", END)
            else:
                print("mcp client no initialize, but moving on anyways")
    except KeyboardInterrupt as ke:
        print("Keyboard Interrupt", ke)


class Chat(BaseModel):
    text: str
    search: Union[bool, None] = None


class MailPrompt(BaseModel):
    prompt: str
    token: str


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


def get_message_body(message):
    payload = message.get("payload", {})
    parts = payload.get("parts", [])

    def extract_part(parts):
        for part in parts:
            _ = part.get("mimeType")
            body = part.get("body")
            data = body.get("data")
            if data:
                enc_data = base64.urlsafe_b64decode(data).decode(encoding="utf-8")
                return enc_data
            elif part.get("parts"):
                return extract_part(part["parts"])

            return None

    if parts:
        return extract_part(parts)
    data = payload.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode

    return "[No message body found]"


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


@app.post("/gmail")
async def do_gmail(prompt: MailPrompt):
    # service = build("gmail", "v1", credentials=Credentials(token=prompt.token))
    # Important comment below
    # pylint: disable=maybe-no-member
    # pylint: disable:R1710
    # messages = (
    #     service.users()
    #     .messages()
    #     .list(userId="allyearmustobey@gmail.com", maxResults=5)
    #     .execute()
    # )

    # msg_id = messages["messages"][0]["id"]
    # msg = (
    #     service.users()
    #     .messages()
    #     .get(userId="allyearmustobey@gmail.com", id=msg_id, format="full")
    #     .execute()
    # )
    # print(messages)
    # pprint.pprint(msg)

    # return get_message_body(msg)

    gmail_prompt_template = ChatPromptTemplate.from_template(
        """
You are an AI assistant connected to the user's Gmail account via secure OAuth authentication. 
Your job is to perform actions on the user's Gmail account as requested using the provided token.

Available action: reading Gmail messages.

When the user asks to check, read, summarize, or fetch their emails, use the appropriate tool to retrieve the messages.

User Request:
{input}

Token: {token}                                                                                                                     
"""
    )

    chain = gmail_prompt_template | App
    return chain.invoke(
        {"input": prompt.prompt, "token": prompt.token},
        {"configurable": {"thread_id": "1234"}},
    )
    # template = gmail_prompt_template.invoke(
    #     {"input": prompt.prompt, "token": prompt.token}
    # )

    # res = await App.ainvoke(
    #     {"messages": template.}, {"configurable": {"thread_id": "1234"}}
    # )
    # return res
    # return res


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
    # print(App.get_state({"configurable": {"thread_id": "1234"}}))
    return res["messages"][-1].content


print("running")

if __name__ == "__main__":
    subprocess.run(["fastapi", "dev", "main.py"])
    p = subprocess.Popen(["fastapi", "dev", "main.py"])
    p.communicate()
