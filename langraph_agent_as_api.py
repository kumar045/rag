import asyncio
import os
import boto3 # Import boto3 for AWS S3 interaction
from io import BytesIO # Import BytesIO for S3 upload
import tempfile # Import tempfile for temporary file handling

from typing import Annotated, TypedDict, Sequence, List, Any, Optional

# import psutil # Not used
# import requests # Used by tools in server.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from motor.motor_asyncio import AsyncIOMotorClient # Import for async MongoDB client

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
import uvicorn

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
print("Langfuse integration has been removed.")

# --- Configuration for Databases and Server ---
MONGO_URI = os.getenv("MONGO_DATABASE_URL", "mongodb://localhost:27017/")
MONGO_LTM_DB_NAME = "app_ltm_db"
MONGO_LTM_COLLECTION_NAME = "user_profile_memories"
MONGO_STM_DB_NAME = "app_stm_db"
MONGO_STM_COLLECTION_NAME = "conversation_threads"

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME") # Get S3 bucket name from environment variables

MCP_SERVER_TARGET_PORT = 8000 # Port where standalone server.py is expected to run
print(f"INFO: MCP Client will target standalone server on port: {MCP_SERVER_TARGET_PORT}")

# --- FastAPI App Instance ---
app = FastAPI(title="LangGraph Agent API", version="1.0.0")

# Custom exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Request validation error: {exc.errors()}")

    # Clean up the exception to prevent binary data serialization issues
    cleaned_errors = []
    for error in exc.errors():
        cleaned_error = error.copy()
        # Remove or sanitize any binary data that might cause encoding issues
        if 'input' in cleaned_error:
            del cleaned_error['input']
        if 'ctx' in cleaned_error and isinstance(cleaned_error['ctx'], dict):
            # Remove any context that might contain binary data
            cleaned_error['ctx'] = {k: v for k, v in cleaned_error['ctx'].items()
                                  if not isinstance(v, bytes)}
        cleaned_errors.append(cleaned_error)

    # Explicitly set body to None to prevent decoding issues during error serialization
    exc.body = None

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": cleaned_errors}
    )

# Custom exception handler for generic Exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    print(f"An unhandled error occurred: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )

# --- MongoDB Client Initialization ---
mongo_client: Optional[MongoClient] = None
async_mongo_client: Optional[AsyncIOMotorClient] = None # Add async client
# Note: MongoDB clients will be initialized in the startup event

# --- LTM MongoDB Utilities ---
def get_ltm_collection() -> Optional[Collection]:
    if not mongo_client: return None
    return mongo_client[MONGO_LTM_DB_NAME][MONGO_LTM_COLLECTION_NAME]

def init_ltm_db_mongo():
    global mongo_client
    if not mongo_client:
        try:
            uri_to_print = MONGO_URI[:MONGO_URI.find('@')] if '@' in MONGO_URI else MONGO_URI[:20]
            print(f"Attempting to connect to MongoDB (LTM init) with URI: {uri_to_print}... (credentials masked)")
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command('ping')
            print("MongoDB connected successfully (LTM init).")
        except Exception as e:
            print(f"ERROR: MongoDB connection failed during LTM init: {e}")
            mongo_client = None
            return

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
compiled_graph: Optional[Any] = None

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
    print(f"[Node] LTM Novelty: LLM check. Score: {res.score}"); state["is_new_info_for_ltm"]="No"; return state
def ltm_storer_node(state:AgentState) -> AgentState:
    user_id=state["user_id"]; info=state.get("personal_info_extracted","")
    if state.get("is_new_info_for_ltm","").lower()=="yes" and info and info!="No new PI found.":
        append_to_ltm_list_mongo(user_id,"user_facts_preferences_list",info)
    else: print(f"[Node] LTM Storer: Skipped for user '{user_id}'.")
    return state

AGENT_LTM_PROMPT_TEMPLATE = """You are a helpful and friendly assistant.
You have access to the following tools to help the user:
{tool_descriptions}

**VERY IMPORTANT INSTRUCTION FOR TOOL USE:**s

If a user asks a question about a specific document (like a PDF or image) AND provides both a file path AND a clear question about that document (e.g., "analyze 'my_document.pdf' and tell me X", "what's in 'image.png' at './path/to/image.png' related to Y", or if the message starts with "Tool to use: query_solver. document_path: '...'. query: '...'."), you MUST use the 'query_solver' tool.
Ensure you extract:
1. The `document_path` from the user's message.
2. The user's question about the document as the `query` argument for the tool.
If both are present, call the 'query_solver' tool immediately with these arguments. Do not ask for clarification if a path and a question about the document are already provided or if the message explicitly states the tool and its arguments.
For other tasks, use other tools if appropriate. Explain tool use and results.

Long-Term Memory about the user:
{long_term_memories_for_prompt}

Use all this information to personalize responses. If no tool is needed for other types of queries, chat normally.
Provide a comprehensive, direct answer to the user's last question.""" # UPDATED PROMPT

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

# Enhanced custom exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Request validation error: {exc.errors()}")

    # Clean up the exception to prevent binary data serialization issues
    cleaned_errors = []
    for error in exc.errors():
        cleaned_error = error.copy()
        # Remove or sanitize any binary data that might cause encoding issues
        if 'input' in cleaned_error:
            del cleaned_error['input']
        if 'ctx' in cleaned_error and isinstance(cleaned_error['ctx'], dict):
            # Remove any context that might contain binary data
            cleaned_error['ctx'] = {k: v for k, v in cleaned_error['ctx'].items()
                                  if not isinstance(v, bytes)}
        cleaned_errors.append(cleaned_error)

    # Explicitly set body to None to prevent decoding issues during error serialization
    exc.body = None

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": cleaned_errors}
    )

# --- FastAPI App Definition ---
app = FastAPI(title="LangGraph Agent API", version="1.0.0")

# --- Request/Response Models for API ---
class AgentInvokeRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    document_path: Optional[str] = None

class UploadFilesResponse(BaseModel):
    filenames: List[str]
    message: str
    retrieved_filepath: Optional[str] = None # Add field for retrieved file path

class AgentInvokeResponse(BaseModel):
    ai_response: str
    thread_id: str

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    global mongo_client, compiled_graph, async_mongo_client

    if not mongo_client:
        try:
            uri_to_print = MONGO_URI[:MONGO_URI.find('@')] if '@' in MONGO_URI else MONGO_URI[:20]
            print(f"Attempting to connect to MongoDB (FastAPI startup) with URI: {uri_to_print}... (credentials masked)")
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            mongo_client.admin.command('ping')
            print("MongoDB connected successfully (FastAPI startup).")
        except Exception as e:
            print(f"ERROR: MongoDB connection failed during FastAPI startup: {e}")
            mongo_client = None

    # Initialize async client
    if MONGO_URI:
        try:
            async_mongo_client = AsyncIOMotorClient(MONGO_URI)
            await async_mongo_client.admin.command('ping')
            print("Async MongoDB client connected successfully (FastAPI startup).")
        except Exception as e:
            print(f"ERROR: Async MongoDB connection failed during FastAPI startup: {e}")
            async_mongo_client = None


    if mongo_client: init_ltm_db_mongo()
    else: print("Skipping LTM DB init (no Mongo client during startup).")

    await async_setup_mcp_client_components()

    workflow = StateGraph(AgentState)
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

    checkpointer = None
    if mongo_client and LANGGRAPH_MONGODB_AVAILABLE and MongoDBSaver is not None:
        print(f"Attempting MongoDBSaver for checkpoints in db '{MONGO_STM_DB_NAME}'.")
        try:
            checkpointer = MongoDBSaver(client=mongo_client,database_name=MONGO_STM_DB_NAME,collection_name=MONGO_STM_COLLECTION_NAME)
            print(f"DEBUG: Checkpointer type: {type(checkpointer)}, Is MongoDBSaver: {isinstance(checkpointer, MongoDBSaver)}")
        except Exception as e: print(f"ERROR: MongoDBSaver init failed: {e}"); checkpointer = None
    if not checkpointer:
        print("Using MemorySaver for checkpoints (STM not persistent)."); checkpointer = MemorySaver()

    print(f"DEBUG: Final checkpointer type for FastAPI app: {type(checkpointer)}")
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    print("Graph compiled successfully for FastAPI app.")

@app.on_event("shutdown")
def shutdown_event():
    global mongo_client, async_mongo_client
    if mongo_client:
        mongo_client.close()
        print("MongoDB client closed during shutdown.")
    if async_mongo_client:
        async_mongo_client.close()
        print("Async MongoDB client closed during shutdown.")
    print("FastAPI application shutdown.")

# --- API Endpoint ---
@app.post("/invoke_agent", response_model=AgentInvokeResponse)
async def invoke_agent_endpoint(request_data: AgentInvokeRequest):
    global compiled_graph
    if compiled_graph is None:
        print("ERROR: Graph not compiled. Check startup event.")
        raise HTTPException(status_code=500, detail="Graph not available")
    try:
        config = {
            "configurable": {
                "thread_id": request_data.thread_id,
                "user_id": request_data.user_id
            }
        }

        user_message_content = request_data.message
        # MODIFIED: Construct a more direct message if document_path is provided
        if request_data.document_path:
            user_message_content = (
                f"Tool to use: query_solver. "
                f"document_path: '{request_data.document_path}'. "
                f"query: '{request_data.message}'."
            )
            print(f"Formatted message for RAG tool call: {user_message_content}")

        graph_input = {"messages": [HumanMessage(content=user_message_content)]}

        print(f"Invoking agent for user_id: {request_data.user_id}, thread_id: {request_data.thread_id}")

        graph_response = await compiled_graph.ainvoke(graph_input, config=config)

        ai_message_content = "Error: No AI response found in graph output."
        if graph_response and graph_response.get("messages"):
            last_message = graph_response["messages"][-1]
            if isinstance(last_message, AIMessage):
                ai_message_content = last_message.content
            else:
                ai_message_content = f"Unexpected last message type: {type(last_message).__name__}. Content: {getattr(last_message, 'content', 'N/A')}"

        return AgentInvokeResponse(
            ai_response=str(ai_message_content),
            thread_id=request_data.thread_id
        )
    except Exception as e:
        print(f"Error in /invoke_agent endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_files", response_model=UploadFilesResponse)
async def upload_files_endpoint(
    user_id: str = Form(..., description="User identifier"),
    thread_id: str = Form(..., description="Thread identifier"),
    message: str = Form(..., description="Message content"),
    file: Optional[UploadFile] = File(None, description="Optional file to upload"),
):
    """
    Upload files endpoint that handles optional file uploads or retrieves from MongoDB.

    - **user_id**: Required user identifier
    - **thread_id**: Required thread identifier
    - **message**: Required message content
    - **file**: Optional file to upload
    - **fileId**: Optional file ID from MongoDB
    """

    # os.makedirs(upload_dir, exist_ok=True) # Removed local directory creation
    uploaded_filename = None
    retrieved_filepath = None
    message_text = ""

    print(f"Received upload request for user_id: {user_id}, thread_id: {thread_id}, message: {message}")

    uploaded_filename = None
    retrieved_filepath = None
    message_text = ""

    if file and file.filename and file.filename.strip():
        if S3_BUCKET_NAME is None:
             raise HTTPException(status_code=500, detail="S3_BUCKET_NAME environment variable not set")

        try:
            uploaded_filename = file.filename
            s3_key = uploaded_filename # Use filename as S3 key
            retrieved_filepath = f"s3://{S3_BUCKET_NAME}/{s3_key}" # Set retrieved_filepath to S3 URI

            # Read file content
            content = await file.read()

            # Upload file to S3
            s3_client = boto3.client("s3")
            # Use upload_fileobj for streaming upload without saving to disk first
            file_content_io = BytesIO(content)
            s3_client.upload_fileobj(file_content_io, S3_BUCKET_NAME, s3_key)

            print(f"Successfully uploaded file to S3: {uploaded_filename} for user {user_id}")
            message_text = f"Successfully uploaded file '{uploaded_filename}' to S3 bucket '{S3_BUCKET_NAME}' for user {user_id} in thread {thread_id}."

        except Exception as e:
            print(f"Error uploading file {uploaded_filename} to S3: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading file {uploaded_filename} to S3: {str(e)}"
            )
    else:
        print(f"No file uploaded for user {user_id} in thread {thread_id}.")
        message_text = f"No file uploaded. Received form data for user {user_id} in thread {thread_id}."

    return UploadFilesResponse(
        filenames=[uploaded_filename] if uploaded_filename else [],
        message=message_text,
        retrieved_filepath=retrieved_filepath # This will be the path from DB, not the local downloaded path
    )

# Alternative endpoint if you want to handle both file upload and agent invocation
@app.post("/upload_and_process", response_model=AgentInvokeResponse)
async def upload_and_process_endpoint(
    user_id: str = Form(...),
    thread_id: str = Form(...),
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """
    Upload a file or provide a fileId and immediately process it with the agent.
    """
    document_path = None # This will be the path to the S3 URI

    if file and file.filename and file.filename.strip():
        if S3_BUCKET_NAME is None:
             raise HTTPException(status_code=500, detail="S3_BUCKET_NAME environment variable not set")

        try:
            uploaded_filename = file.filename
            s3_key = uploaded_filename # Use filename as S3 key
            document_path = f"s3://{S3_BUCKET_NAME}/{s3_key}" # Set document_path to S3 URI

            # Read file content
            content = await file.read()

            # Upload file to S3
            s3_client = boto3.client("s3")
            from io import BytesIO
            file_content_io = BytesIO(content)
            s3_client.upload_fileobj(file_content_io, S3_BUCKET_NAME, s3_key)

            print(f"Uploaded file to S3: {uploaded_filename}")

            # Download the uploaded file to a temporary local path for processing
            try:
                s3_client = boto3.client("s3")
                # Create a temporary file to download into
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1]) as tmp_file:
                    s3_client.download_fileobj(S3_BUCKET_NAME, s3_key, tmp_file)
                    document_path = tmp_file.name # Set document_path to the temporary file path
                print(f"Downloaded S3 file for processing to temporary path: {document_path}")
            except Exception as e:
                print(f"Error downloading uploaded file from S3 for processing: {e}")
                raise HTTPException(status_code=500, detail=f"Error downloading uploaded file from S3: {str(e)}")

        except Exception as e:
            print(f"Error uploading file {uploaded_filename} to S3 for processing: {e}")
            raise HTTPException(status_code=500, detail=f"Error uploading file {uploaded_filename} to S3: {str(e)}")
    else:
        # If no file is provided, proceed without a document_path
        print("No file uploaded for processing.")
        document_path = None # Ensure document_path is None if no file

    # Create agent request
    agent_request = AgentInvokeRequest(
        user_id=user_id,
        thread_id=thread_id,
        message=message,
        document_path=document_path # Pass the local downloaded path
    )

    # Process with agent
    return await invoke_agent_endpoint(agent_request)


# --- Main Execution Block for FastAPI App (No direct test invocations here) ---
if __name__ == "__main__":
    try:
        print("Starting FastAPI server for LangGraph agent...")
        fastapi_port = int(os.getenv("APP_PORT", 8080))
        fastapi_host = os.getenv("APP_HOST", "127.0.0.1")
        uvicorn.run(app, host=fastapi_host, port=fastapi_port)
    except Exception as e:
        print(f"Error setting up or running FastAPI server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application (langraph_agent_as_api.py FastAPI server) exiting its __main__ block.")
        if mongo_client:
            mongo_client.close()
            print("MongoDB client closed (extra check).")
        if async_mongo_client:
            async_mongo_client.close()
            print("Async MongoDB client closed (extra check).")
