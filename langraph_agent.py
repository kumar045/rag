import asyncio
import os

from typing import Annotated, TypedDict, Sequence, List, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

# LangChain & LangGraph Core
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
# from langchain_core.tracers.langfuse import LangfusePlusCallbackHandler # Langfuse Removed
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # Default STM checkpointer

# LLMs (Using Google Generative AI)
from langchain_google_genai import ChatGoogleGenerativeAI

# MCP Client (Server is now separate)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import Tool

# Agent Creation
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

# MongoDB
from pymongo import MongoClient, errors as pymongo_errors
from pymongo.collection import Collection


LANGGRAPH_MONGODB_AVAILABLE = False
MongoDBSaver = None

# 0. Environment Setup
load_dotenv()
DEFAULT_USER_ID = "default-user"

# --- Langfuse Initialization REMOVED ---
print("Langfuse integration has been removed.")


# --- Configuration for Databases and Server ---
MONGO_URI = os.getenv("MONGO_DATABASE_URL", "mongodb://localhost:27017/")
MONGO_LTM_DB_NAME = "app_ltm_db"
MONGO_LTM_COLLECTION_NAME = "user_profile_memories"
MONGO_STM_DB_NAME = "app_stm_db"
MONGO_STM_COLLECTION_NAME = "conversation_threads"

MCP_SERVER_TARGET_PORT = 8000 # Port where standalone server.py is expected to run
print(f"INFO: MCP Client will target standalone server on port: {MCP_SERVER_TARGET_PORT}")

# --- MongoDB Client Initialization ---
mongo_client: Optional[MongoClient] = None
try:
    uri_to_print = MONGO_URI[:MONGO_URI.find('@')] if '@' in MONGO_URI else MONGO_URI[:20]
    print(f"Attempting to connect to MongoDB with URI: {uri_to_print}... (credentials masked)")
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    print("Successfully connected to MongoDB.")
except pymongo_errors.ConfigurationError as e:
    print(f"ERROR: MongoDB ConfigurationError: {e}"); mongo_client = None
except pymongo_errors.ConnectionFailure as e:
    print(f"ERROR: Could not connect to MongoDB."); print(f"MongoDB ConnectionFailure: {e}"); mongo_client = None
except Exception as e:
    print(f"ERROR: Unexpected error during MongoDB client init: {e}"); mongo_client = None
if not mongo_client:
    print("WARNING: Proceeding without MongoDB. LTM and persistent STM will not work.")

# --- LTM MongoDB Utilities ---
def get_ltm_collection() -> Optional[Collection]:
    if not mongo_client: return None
    return mongo_client[MONGO_LTM_DB_NAME][MONGO_LTM_COLLECTION_NAME]

def init_ltm_db_mongo():
    if not mongo_client: print("LTM: MongoDB client N/A. Skipping DB init."); return
    try:
        coll = get_ltm_collection()
        if coll is not None: print(f"LTM: MongoDB collection '{MONGO_LTM_COLLECTION_NAME}' ready.")
        else: print("LTM: Failed to get LTM collection.")
    except Exception as e: print(f"LTM: Error during MongoDB LTM init: {e}")

def append_to_ltm_list_mongo(user_id: str, list_key: str, item: str):
    coll = get_ltm_collection();
    if coll is None: print(f"LTM: No DB. Cannot append for user '{user_id}'."); return
    try:
        coll.update_one({"_id": user_id}, {"$addToSet": {list_key: item}, "$setOnInsert": {"_id": user_id}}, upsert=True)
        print(f"LTM (Mongo): Appended/Ensured '{item}' in '{list_key}' for user '{user_id}'.")
    except pymongo_errors.PyMongoError as e: print(f"LTM (Mongo) DB Error appending for {user_id}: {e}")

def get_ltm_list_mongo(user_id: str, list_key: str) -> List[str]:
    coll = get_ltm_collection();
    if coll is None: return []
    try:
        doc = coll.find_one({"_id": user_id})
        return doc.get(list_key, []) if doc and isinstance(doc.get(list_key), list) else []
    except pymongo_errors.PyMongoError as e: print(f"LTM (Mongo) DB Error getting list for {user_id}: {e}"); return []

# --- LangGraph Agent State and Pydantic Models ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]; user_id: str
    personal_info_detected: str; personal_info_extracted: str
    is_new_info_for_ltm: str; collected_long_term_memories: str
class GradeQuestion(BaseModel): score: str = Field(description="Is info present? 'Yes'/'No'.")
class InfoNoveltyGrade(BaseModel): score: str = Field(description="Is new info unique? 'Yes'/'No'.")
class NoArgsPydanticModel(BaseModel): pass

# --- Globals ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
MCP_SERVER_TOOLS: List[Tool] = []
agent_executor_instance: Optional[AgentExecutor] = None

# --- LangGraph Nodes ---
def set_user_id_node(state:AgentState, config:RunnableConfig) -> AgentState:
    state["user_id"]=config["configurable"].get("user_id",DEFAULT_USER_ID); print(f"[Node] User ID: {state['user_id']}"); return state
def retrieve_ltm_node(state:AgentState) -> AgentState:
    user_id=state["user_id"]; print(f"[Node] Retrieve LTM for {user_id}")
    facts=get_ltm_list_mongo(user_id,"user_facts_preferences_list")
    state["collected_long_term_memories"]="\n".join(facts) if facts else "No LTM found for you yet."
    print(f"  Retrieved LTM (Mongo): {state['collected_long_term_memories'][:200]}..."); return state
def pi_classifier_node(state:AgentState) -> AgentState:
    msg=state["messages"][-1].content; sys_prompt="Classifier: New PI in LAST msg? Yes/No."
    prompt=ChatPromptTemplate.from_messages([("system",sys_prompt),("human","{message}")]); chain=prompt|llm.with_structured_output(GradeQuestion)
    res=chain.invoke({"message":msg}); print(f"[Node] PI Classifier: {res.score}"); state["personal_info_detected"]=res.score.strip(); return state
def pi_extractor_node(state:AgentState) -> AgentState:
    msg=state["messages"][-1].content
    sys_prompt="Extract new PI from LAST msg as facts. E.g., User: \"I'm John, live in Seattle.\" -> \"User is John. User lives in Seattle.\". If none: \"No new PI found.\"";
    prompt=ChatPromptTemplate.from_messages([("system",sys_prompt),("human","Input: {message}")]); chain=prompt|llm
    info=chain.invoke({"message":msg}); print(f"[Node] PI Extractor: {info.content.strip()}"); state["personal_info_extracted"]=info.content.strip(); return state
def ltm_novelty_node(state:AgentState) -> AgentState:
    user_id=state["user_id"]; new_info=state.get("personal_info_extracted","")
    if not new_info or new_info=="No new PI found.": print("[Node] LTM Novelty: No new info."); state["is_new_info_for_ltm"]="No"; return state
    existing=get_ltm_list_mongo(user_id,"user_facts_preferences_list")
    if new_info in existing: print(f"[Node] LTM Novelty: Exact LTM duplicate. No"); state["is_new_info_for_ltm"]="No"; return state
    existing_str="\n".join(existing) if existing else "No LTM."; sys_msg="Is 'New info' new vs 'Existing LTM'? If new fact/pref, 'Yes'. Else 'No'."
    tmpl="New info:\n{new_info}\n\nExisting LTM:\n{existing_str}\n\nNew for LTM? Yes/No."
    prompt=ChatPromptTemplate.from_messages([("system",sys_msg),("human",tmpl.format(new_info=new_info,existing_str=existing_str))])
    chain=prompt|llm.with_structured_output(InfoNoveltyGrade); res=chain.invoke({})
    print(f"[Node] LTM Novelty: LLM check. Score: {res.score}"); state["is_new_info_for_ltm"]=res.score.strip(); return state
def ltm_storer_node(state:AgentState) -> AgentState:
    user_id=state["user_id"]; info=state.get("personal_info_extracted","")
    if state.get("is_new_info_for_ltm","").lower()=="yes" and info and info!="No new PI found.":
        append_to_ltm_list_mongo(user_id,"user_facts_preferences_list",info)
    else: print(f"[Node] LTM Storer: Skipped for user '{user_id}'.")
    return state

AGENT_LTM_PROMPT_TEMPLATE = """You are a helpful and friendly assistant.
You have access to the following tools to help the user:
{tool_descriptions}

**VERY IMPORTANT INSTRUCTION FOR TOOL USE:**
If a user asks a question about a specific document (like a PDF or image) AND provides both a file path AND a clear question about that document (e.g., "analyze 'my_document.pdf' and tell me X", "what's in 'image.png' at './path/to/image.png' related to Y"), you MUST use the 'query_solver' tool.
Ensure you extract:
1. The `document_path` from the user's message.
2. The user's question about the document as the `query` argument for the tool.
If both are present, call the 'query_solver' tool immediately with these arguments. Do not ask for clarification if a path and a question about the document are already provided.
For other tasks, use other tools if appropriate. Explain tool use and results.

Long-Term Memory about the user:
{long_term_memories_for_prompt}

Use all this information to personalize responses. If no tool is needed for other types of queries, chat normally.
Provide a comprehensive, direct answer to the user's last question."""

def create_agent_exec(tools:List[Tool]):
    llm_tools=llm.bind_tools(tools)
    runnable=({"messages":lambda x:x["messages"],"long_term_memories_for_prompt":lambda x:x["long_term_memories_for_prompt"],"tool_descriptions":lambda x:x["tool_descriptions"],"agent_scratchpad":lambda x:format_to_tool_messages(x["intermediate_steps"])}|ChatPromptTemplate.from_messages([("system",AGENT_LTM_PROMPT_TEMPLATE),MessagesPlaceholder(variable_name="messages"),MessagesPlaceholder(variable_name="agent_scratchpad")])|llm_tools|ToolsAgentOutputParser())
    return AgentExecutor(agent=runnable,tools=tools,verbose=True,handle_parsing_errors="ERROR: Agent Parsing Error")

async def agent_call_node(state:AgentState)->AgentState:
    global agent_executor_instance; print("--- [Node] Calling Agent ---")
    ltm=state.get("collected_long_term_memories","No LTM."); msgs=list(state["messages"])
    if agent_executor_instance is None:
        print("CRIT: Agent exec None. Fallback."); prompt=ChatPromptTemplate.from_messages([("system",f"Assistant. User LTM: {ltm}"),MessagesPlaceholder(variable_name="messages")]); chain=prompt|llm
        response = await chain.ainvoke({"messages":msgs})
        state["messages"]+=[AIMessage(content=str(response.content if hasattr(response, 'content') else response))]; return state
    tool_descs="\n".join([f"- {t.name}: {t.description}" for t in MCP_SERVER_TOOLS]) if MCP_SERVER_TOOLS else "No tools."
    agent_in={"messages":msgs,"long_term_memories_for_prompt":ltm,"tool_descriptions":tool_descs}
    try: 
        response = await agent_executor_instance.ainvoke(agent_in)
        final_msg=AIMessage(content=str(response.get("output","Error.")))
    except Exception as e: print(f"ERR AgentExec: {e}"); final_msg=AIMessage(content="Error processing request.")
    state["messages"]+=[final_msg]; return state

# --- LangGraph Workflow Definition ---
workflow=StateGraph(AgentState)
workflow.add_node("set_user_id",set_user_id_node); workflow.add_node("retrieve_ltm",retrieve_ltm_node)
workflow.add_node("classify_pi",pi_classifier_node); workflow.add_node("extract_pi",pi_extractor_node)
workflow.add_node("check_ltm_novelty",ltm_novelty_node); workflow.add_node("store_ltm",ltm_storer_node)
workflow.add_node("call_agent",agent_call_node)
workflow.set_entry_point("set_user_id"); workflow.add_edge("set_user_id","retrieve_ltm")
workflow.add_edge("retrieve_ltm","classify_pi")
def route_pi_classify(s:AgentState): return "extract_pi" if s["personal_info_detected"].lower()=="yes" else "call_agent"
workflow.add_conditional_edges("classify_pi",route_pi_classify)
workflow.add_edge("extract_pi","check_ltm_novelty")
def route_ltm_novel(s:AgentState): return "store_ltm" if s.get("is_new_info_for_ltm","").lower()=="yes" else "call_agent"
workflow.add_conditional_edges("check_ltm_novelty",route_ltm_novel)
workflow.add_edge("store_ltm","call_agent"); workflow.add_edge("call_agent",END)

# --- Checkpointer Setup ---
checkpointer = None
if mongo_client and LANGGRAPH_MONGODB_AVAILABLE and MongoDBSaver is not None:
    print(f"Attempting MongoDBSaver for checkpoints in db '{MONGO_STM_DB_NAME}'. Ensure 'langgraph-mongodb' installed.")
    try:
        checkpointer = MongoDBSaver(client=mongo_client,database_name=MONGO_STM_DB_NAME,collection_name=MONGO_STM_COLLECTION_NAME)
        print(f"DEBUG: Checkpointer type: {type(checkpointer)}, Is MongoDBSaver: {isinstance(checkpointer, MongoDBSaver)}")
    except Exception as e: print(f"ERROR: MongoDBSaver init failed: {e}"); checkpointer = None
if not checkpointer:
    print("Using MemorySaver for checkpoints. STM not persistent across restarts."); checkpointer = MemorySaver() # Fallback
print(f"DEBUG: Final checkpointer type being used: {type(checkpointer)}")
graph = workflow.compile(checkpointer=checkpointer)
print("Graph compiled successfully.")

# --- MCP Client Setup ---
async def async_setup_mcp_client_components():
    global MCP_SERVER_TOOLS, agent_executor_instance
    print(f"Attempting to connect to standalone MCP Server on port {MCP_SERVER_TARGET_PORT}...")
    await asyncio.sleep(3) 

    servers_config = {"my_mcp_server": {"transport": "sse", "url": f"http://127.0.0.1:{MCP_SERVER_TARGET_PORT}/sse"}}
    client = MultiServerMCPClient(servers_config); processed_tools: List[Tool] = []
    try:
        raw_tools: List[Tool] = await client.get_tools()
        print(f"Fetched {len(raw_tools)} raw tools from MCP server (port {MCP_SERVER_TARGET_PORT}).")
        if raw_tools:
            for i, tool_data in enumerate(raw_tools):
                print(f"  Raw Tool {i}: {tool_data.name}, Original ArgsSchema type: {type(tool_data.args_schema)}")
                current_args_schema = tool_data.args_schema; final_schema_for_tool: Optional[Type[BaseModel]] = None

                if current_args_schema is None or \
                   (isinstance(current_args_schema, dict) and \
                    not current_args_schema.get('properties') and \
                    not any(isinstance(v,dict) and v.get('type') for v in current_args_schema.values())):
                    final_schema_for_tool = NoArgsPydanticModel
                    print(f"    Tool {tool_data.name} has no arguments. Using NoArgsPydanticModel.")
                elif isinstance(current_args_schema, type) and issubclass(current_args_schema, BaseModel):
                    final_schema_for_tool = current_args_schema
                    print(f"    Tool {tool_data.name} already has Pydantic args_schema: {final_schema_for_tool.__name__}")
                elif isinstance(current_args_schema, dict):
                    print(f"    Tool {tool_data.name} args_schema is dict: {current_args_schema}. Attempting dynamic Pydantic model.")
                    properties_dict = current_args_schema.get('properties', None)
                    if properties_dict is None and all(isinstance(v, dict) and 'type' in v for v in current_args_schema.values()):
                        properties_dict = current_args_schema
                    
                    if isinstance(properties_dict, dict) and properties_dict:
                        pydantic_fields = {}
                        for arg_name, arg_props in properties_dict.items():
                            if isinstance(arg_props, dict):
                                arg_type_str=arg_props.get("type","string"); py_type:Any=str
                                if arg_type_str=="integer": py_type=int
                                elif arg_type_str=="number": py_type=float
                                elif arg_type_str=="boolean": py_type=bool
                                elif arg_type_str=="array": py_type=List[Any] 
                                arg_desc=arg_props.get("description")
                                pydantic_fields[arg_name]=(py_type, Field(...,description=arg_desc) if arg_desc else (py_type,...))
                            else: print(f"    Skipping property {arg_name} for {tool_data.name}, value not dict: {arg_props}")
                        if pydantic_fields:
                            try: ModelName=f"{tool_data.name.replace('_',' ').title().replace(' ','')}Args"; final_schema_for_tool=create_model(ModelName,**pydantic_fields); print(f"    Dynamically created Pydantic schema for {tool_data.name}: {ModelName}")
                            except Exception as e_create: print(f"    ERROR creating Pydantic model for {tool_data.name}: {e_create}"); final_schema_for_tool=NoArgsPydanticModel
                        else: print(f"    No valid fields for {tool_data.name}. Using NoArgsPydanticModel."); final_schema_for_tool=NoArgsPydanticModel
                    else: print(f"    Dict args_schema for {tool_data.name} not recognized. Using NoArgsPydanticModel."); final_schema_for_tool=NoArgsPydanticModel
                else: print(f"    WARN: Unknown args_schema for {tool_data.name}: {type(current_args_schema)}"); final_schema_for_tool=NoArgsPydanticModel
                
                if hasattr(tool_data, 'args_schema'): tool_data.args_schema = final_schema_for_tool
                processed_tools.append(tool_data)
            MCP_SERVER_TOOLS = processed_tools
        else: MCP_SERVER_TOOLS = []
        agent_executor_instance = create_agent_exec(MCP_SERVER_TOOLS)
        print(f"AgentExecutor initialized with {len(MCP_SERVER_TOOLS)} processed tools.")
    except Exception as e:
        print(f"ERROR MCP client setup: {e} (Standalone server.py running on port {MCP_SERVER_TARGET_PORT}?)")
        MCP_SERVER_TOOLS = []; agent_executor_instance = create_agent_exec([])
        print("Proceeding without external tools.")

# --- Main Execution Block ---
async def main():
    if mongo_client: init_ltm_db_mongo()
    else: print("Skipping LTM DB init (no Mongo client).")
    
    await async_setup_mcp_client_components()
    
    try:
        img_data=graph.get_graph(xray=True).draw_mermaid_png();open("graph_ltm_mongo_viz.png","wb").write(img_data)
        print("Graph viz saved.")
    except Exception as e: print(f"No graph viz: {e} (Mermaid CLI/API issues?)")

    print("\n\n--- Test Scenario (MongoDB LTM, External MCP Server) ---")
    user_alex_id="user_alex_split_001"; user_bella_id="user_bella_split_002"

    invocation_callbacks = [] # No Langfuse for now

    # Alex - Session 1
    config_alex_s1 = {"configurable": {"thread_id": "alex_split_s1", "user_id": user_alex_id}, "callbacks": invocation_callbacks}
    print(f"\n--- Alex (ID: {user_alex_id}), Session 1 (Thread: alex_split_s1) ---")
    print("\nAlex: Hi! I'm Alex. My favorite color is blue and I enjoy hiking.")
    r1=await graph.ainvoke({"messages":[HumanMessage(content="Hi! I'm Alex. My favorite color is blue and I enjoy hiking.")]}, config=config_alex_s1)
    if r1 and r1.get("messages"): print(f"AI: {r1['messages'][-1].content}")
    
    pdf_document_path_for_tool = "documents/graph_ltm_mongo_viz.png" 
    # Ensure this file exists where server.py is run for the tool to find it
    
    pdf_query_prompt = f"Please analyze the document at '{pdf_document_path_for_tool}' and tell me, what are the key findings?"
    print(f"\nAlex: {pdf_query_prompt}")
    
    r_pdf = await graph.ainvoke({"messages":[HumanMessage(content=pdf_query_prompt)]}, config=config_alex_s1)
    if r_pdf and r_pdf.get("messages"): print(f"AI: {r_pdf['messages'][-1].content}")


    print("\nAlex: What's 75 + 25 using the calculator?")
    r2=await graph.ainvoke({"messages":[HumanMessage(content="What's 75 + 25 using the calculator?")]}, config_alex_s1)
    if r2 and r2.get("messages"): print(f"AI: {r2['messages'][-1].content}")

    # Alex - Session 2
    config_alex_s2 = {"configurable": {"thread_id": "alex_split_s2", "user_id": user_alex_id}, "callbacks": invocation_callbacks}
    print(f"\n--- Alex (ID: {user_alex_id}), Session 2 (Thread: alex_split_s2) ---")
    print("\nAlex: Do you remember my name or my favorite color?")
    r3=await graph.ainvoke({"messages":[HumanMessage(content="Do you remember my name or my favorite color?")]}, config_alex_s2)
    if r3 and r3.get("messages"): print(f"AI: {r3['messages'][-1].content}")
    
    print("\nAlex: Generate a UUID for me please.")
    r4=await graph.ainvoke({"messages":[HumanMessage(content="Generate a UUID for me please.")]}, config_alex_s2)
    if r4 and r4.get("messages"): print(f"AI: {r4['messages'][-1].content}")

    # Bella - Session 1
    config_bella_s1 = {"configurable": {"thread_id": "bella_split_s1", "user_id": user_bella_id}, "callbacks": invocation_callbacks}
    print(f"\n--- Bella (ID: {user_bella_id}), Session 1 (Thread: bella_split_s1) ---")
    print("\nBella: Hello, I'm Bella. I prefer very concise answers, and I work as a journalist.")
    r_b1=await graph.ainvoke({"messages":[HumanMessage(content="Hello, I'm Bella. I prefer concise answers, I am a journalist.")]}, config_bella_s1)
    if r_b1 and r_b1.get("messages"): print(f"AI: {r_b1['messages'][-1].content}")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nExiting...")
    except Exception as e: print(f"Unhandled main exception: {e}"); import traceback; traceback.print_exc()
    finally:
        print("Final cleanup (app.py)...")
        if mongo_client: mongo_client.close(); print("MongoDB client closed.")
        print("Cleanup complete (app.py).")

