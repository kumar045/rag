import os
import base64
import uuid
import requests
from datetime import datetime, timedelta
from io import BytesIO
from pdf2image import convert_from_path
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp.server.fastmcp import FastMCP


# ------------------------
# LLM Setup (Gemini)
# ------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# ------------------------
# Utility: Encode images or PDF page as Data URI
# ------------------------

def encode_image_to_data_uri(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        ext = image_path.split('.')[-1]
        mime = f"image/{'jpeg' if ext.lower() in ['jpg', 'jpeg'] else ext.lower()}"
        return f"data:{mime};base64,{encoded}"

def pdf_page_to_data_uri(pdf_path, page=0):
    images = convert_from_path(pdf_path, first_page=page+1, last_page=page+1)
    img = images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def create_message(file_path, prompt="What's in this file?"):
    ext = file_path.split('.')[-1].lower()
    if ext in ["jpg", "jpeg", "png"]:
        data_uri = encode_image_to_data_uri(file_path)
    elif ext == "pdf":
        data_uri = pdf_page_to_data_uri(file_path, page=0)
    else:
        raise ValueError("Unsupported file type")
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": data_uri},
        ]
    )

# ------------------------
# Create Server
# ------------------------

mcp_server = FastMCP("MyStandaloneToolServerWithRAG_DirectSDK")

# ------------------------
# Utility Tools
# ------------------------

@mcp_server.tool(name="basic_calculator", description="Performs basic arithmetic.")
def basic_calculator(operation: str, a: float, b: float) -> float:
    if operation == "add": return a + b
    elif operation == "subtract": return a - b
    elif operation == "multiply": return a * b
    elif operation == "divide":
        if b == 0: raise ZeroDivisionError("Cannot divide by zero.")
        return a / b
    else: raise ValueError(f"Unknown operation: {operation}")

@mcp_server.tool(name="weather_lookup", description="Fetches current weather for a city.")
def weather_lookup(city: str) -> str:
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error fetching weather: {e}"

@mcp_server.tool(name="uuid_generator", description="Generates a random UUID.")
def uuid_generator() -> str:
    return str(uuid.uuid4())

@mcp_server.tool(name="random_joke", description="Returns a random joke.")
def random_joke() -> str:
    url = "https://official-joke-api.appspot.com/random_joke"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        return f"{data['setup']} - {data['punchline']}"
    except Exception as e:
        return f"Failed to fetch joke: {str(e)}"

@mcp_server.tool(name="date_calculator", description="Adds/subtracts days from a date.")
def date_calculator(start_date: str, days: int) -> str:
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")

# ------------------------
# Document Q&A Tool using Gemini
# ------------------------

@mcp_server.tool(name="query_solver", description="Answers a query based on PDF/image file content.")
def answer_document_query(document_path: str, query: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    print(document_path)
    try:
        msg = create_message(document_path, prompt=query)
        return llm.invoke([msg]).content
    except Exception as e:
        return f"Error processing document: {str(e)}"

# ------------------------
# Server Entry Point
# ------------------------

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 8000))
    server_host = "127.0.0.1"
    print(f"Starting MCP server at http://{server_host}:{server_port}")
    mcp_server.run(transport="sse")
