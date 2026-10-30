"""
Business logic for AI agent with knowledge base integration.
"""
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent, ToolCall, ToolCallResult
from llama_index.core.llms import ChatMessage, MessageRole

load_dotenv()

# 1. Initialize LLM and set it globally so RetrieverQueryEngine uses it too
llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
Settings.llm = llm

# 2. Initialize AWS Bedrock Knowledge Base Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=os.getenv("BEDROCK_KNOWLEDGE_BASE_ID"),
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}},
)

# 3. Create Query Engine Tool
_knowledge_base_tool = QueryEngineTool.from_defaults(
    query_engine=RetrieverQueryEngine.from_args(retriever=retriever, llm=llm),
    name="amazon_knowledge_base",
    description=(
        "A vector database of knowledge about EU social media analysis for different countries. "
        "Always use this tool to look up country data or social media metrics."
    ),
)

# 4. Initialize ReActAgent
agent = ReActAgent(
    tools=[_knowledge_base_tool],
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are a helpful AI assistant with access to a vector database of knowledge about EU social media data. "
        "When users ask questions about countries or their social media engagement data, "
        "you MUST use the 'amazon_knowledge_base' tool to retrieve accurate information. "
        "Only reply using facts from the knowledge base. Respond in concise English. "
        "If the tool returns no relevant information, state that you do not know."
    ),
)


async def get_agent_response(message: str, chat_history: list) -> str:
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(ChatMessage(role=MessageRole.USER, content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=msg["content"]))

    # Run the workflow handler so we can log tool calls in the terminal
    handler = agent.run(user_msg=message, chat_history=messages)

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f"\n[AWS KB REQUEST] Calling tool: {event.tool_name} | Args: {event.tool_kwargs}")
        elif isinstance(event, ToolCallResult):
            print(f"[AWS KB RESPONSE] Retrieved from Bedrock KB:\n{event.tool_output}\n")

    response = await handler
    return str(response)