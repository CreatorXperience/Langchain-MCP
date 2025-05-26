from langchain.chat_models import init_chat_model
from typing import Union
from fastapi import FastAPI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel

from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.tools import tool as lang_tool
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from bs4 import BeautifulSoup
from email.message import EmailMessage
from googleapiclient.errors import HttpError

import subprocess
import getpass
import os
import dotenv
import json
import sys
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
    llm = init_chat_model(model="llama3-70b-8192", model_provider="groq", api_key=val)


memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(State)


@lang_tool
def draft_email_messages(
    token: Annotated[str, "token or key to authenticate to gmail service"],
    to: Annotated[str, "email message recipient"],
    subject: Annotated[str, "Subject of the message"],
):
    """Draft an email message using the Gmail API"""
    try:
        service = build("gmail", "v1", credentials=Credentials(token=token))

        message = EmailMessage()
        message.set_content("This is Aura drafted Message")
        message["To"] = to
        message["From"] = "allyearmustobey@gmail.com"
        message["Subject"] = subject
        msg_bytes = message.as_bytes()

        encoded_msg = base64.urlsafe_b64encode(msg_bytes).decode()
        create_message = {"message": {"raw": encoded_msg}}

        # Important comment below
        # pylint: disable=maybe-no-member
        # pylint: disable:R1710

        draft = (
            service.users().drafts().create(userId="me", body=create_message).execute()
        )

        msg = f"message with Draft Id: {draft["id"]} drafted successfully"
        print(msg)
        return msg
    except HttpError as error:
        print(f"error occured while drafting message Error-> {error}")


@lang_tool
def read_gmail_messages(
    token: str, task: Annotated[str, "what the user says says you should do"]
):
    """Fetches and reads email messages from a user's Gmail inbox using
    the Gmail API. Returns message content for further processing"""
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
    query_template = ChatPromptTemplate.from_template(
        """ 
    You are an expert email assistant. Your task is to formulate a Gmail-compatible search query string based on the user's request.

Use Gmail's search syntax. You may combine operators when needed. Do not return any explanation—just the Gmail search query string.

Here are the Gmail search operators you may use:

| Operator           | Description                                      | Example                        |
|--------------------|--------------------------------------------------|--------------------------------|
| `from:`            | Sender's email or domain                         | `from:amazon.com`              |
| `to:`              | Recipient's email                                | `to:john@example.com`          |
| `subject:`         | Words in the subject line                        | `subject:invoice`              |
| `label:`           | Messages with a specific label                   | `label:work`                   |
| `is:unread`        | Unread emails only                               | `is:unread`                    |
| `is:read`          | Read emails only                                 | `is:read`                      |
| `has:attachment`   | Emails with attachments                          | `has:attachment`               |
| `newer_than:Xd`    | Emails newer than X days                         | `newer_than:7d`                |
| `older_than:Xd`    | Emails older than X days                         | `older_than:30d`               |
| `filename:`        | Emails with specific file types                  | `filename:pdf`                 |
| `after:YYYY/MM/DD` | Emails after a specific date                     | `after:2023/01/01`             |
| `before:YYYY/MM/DD`| Emails before a specific date                    | `before:2023/01/31`            |

Return only the Gmail search query that matches the user's request.

Example:
User prompt: *"Find unread emails from Google with PDF attachments in the last 3 days"*
Your output: `from:google.com is:unread has:attachment filename:pdf newer_than:3d`

---

Now process this prompt:
{task}

    """
    )
    chain = query_template | llm_with_tools
    res = chain.invoke({"task": task})
    query = res.content
    print(query)
    messages = (
        service.users()
        .messages()
        .list(userId="allyearmustobey@gmail.com", maxResults=1, q=query)
        .execute()
    )

    msg_obj = []
    msg_data = ""
    print("and here->", messages)
    if "messages" in messages:
        for message in messages["messages"]:
            msg = (
                service.users()
                .messages()
                .get(
                    userId="allyearmustobey@gmail.com", id=message["id"], format="full"
                )
                .execute()
            )
            msg_obj.append(msg)
        print(msg_obj)
        for idx, mg in enumerate(msg_obj):
            data = get_message_body(mg)
            msg_data += f"\nMessage {idx+1}: \n {data}"
    else:
        print("and here->", messages)

    print(msg_data, msg_obj)

    unescaped_data = (
        msg_data.replace("\\r\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
    )
    soup = BeautifulSoup(unescaped_data, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    print(text)
    return text


@asynccontextmanager
async def lifespan(fastapp: FastAPI):
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
    print("here- >", state["messages"])
    result = {"messages": llm_with_tools.invoke(state["messages"])}
    print(result)
    return result


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
    Convert the following text to properly formatted Markdown only if is code, if it's not,  output the text as it is. Do not include any explanations or introductory phrases. Output only the converted Markdown:
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

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}


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
                    draft_email_messages,
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
        return base64.urlsafe_b64decode(data).decode(encoding="utf-8")

    return "[No message body found]"


@app.post("/gmail")
async def do_gmail(prompt: MailPrompt):

    res = await App.ainvoke(
        {"messages": [HumanMessage(prompt.prompt)]},
        {"configurable": {"thread_id": "1234"}},
    )
    return res["messages"][-1].content


@app.post("/talk")
async def talk(chat: Chat):

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
