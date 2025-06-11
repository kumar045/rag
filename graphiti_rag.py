import streamlit as st
import json
import os
import dotenv
import asyncio
import easyocr
from google import genai
from google.genai import types
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from graphiti_core import Graphiti
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode

dotenv.load_dotenv()

# Google API key configuration
api_key = os.getenv("GOOGLE_API_KEY")

os.environ['OPENAI_API_KEY'] = api_key

reader = easyocr.Reader(['en']) # 'en' for English. Add more languages if needed.

class GraphitiSearchResult(BaseModel):
    """Model representing a search result from Graphiti."""
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(description="The factual statement retrieved from the knowledge graph")
    valid_at: Optional[str] = Field(None, description="When this fact became valid (if known)")
    invalid_at: Optional[str] = Field(None, description="When this fact became invalid (if known)")
    source_node_uuid: Optional[str] = Field(None, description="UUID of the source node")

# Initialize Graphiti with Gemini clients
graphiti = Graphiti(
    os.getenv("NEO4J_URI"),
    os.getenv("NEO4J_USER"),
    os.getenv("NEO4J_PASSWORD"),
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.0-flash"
        )
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="embedding-001"
        )
    )
)

# Function to generate content using google-genai
def generate(input_text):
    client = genai.Client(
        api_key=api_key,
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="You are a helpful, expert assistant. Answer concisely and accurately."),
        ],
    ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    generated_content = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        generated_content += chunk.text
    return generated_content


st.title("Graphiti Data Ingestion and Search App")



st.subheader("Data Ingestion")
st.write("Upload a file to ingest data into Graphiti.")

uploaded_file = st.file_uploader("Choose a file", type=["json", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    st.write(f"Processing file: {file_name} ({file_type})")

    try:
        if file_type == "application/json":
            # Process JSON file for bulk ingestion
            data = json.load(uploaded_file)
            if isinstance(data, list):
                st.write(f"Loaded {len(data)} records from JSON file.")
                bulk_episodes = []
                for i, record in enumerate(data):
                    episode_name = f"JSON Ingest - {file_name} - Record {i+1}"
                    episode_body = json.dumps(record) # Ensure it's a string
                    source_type = EpisodeType.json
                    source_description = "Uploaded JSON file record"
                    bulk_episodes.append(
                        EpisodicNode(
                            name=episode_name,
                            content=episode_body,
                            source=source_type,
                            source_description=source_description,
                            valid_at=datetime.now()
                        )
                    )
                
                if bulk_episodes:
                    st.write("Prepared episodes for bulk ingestion.")
                    async def ingest_bulk_data():
                        try:
                            await graphiti.add_episode_bulk(bulk_episodes)
                            st.success(f"Successfully ingested {len(bulk_episodes)} records into Graphiti!")
                        except Exception as e:
                            st.error(f"An error occurred during bulk ingestion: {e}")

                    if st.button("Ingest Bulk Data into Graphiti"):
                        asyncio.run(ingest_bulk_data())

            else:
                st.warning("JSON file does not contain a list of records for bulk ingestion.")
                # Optionally handle single JSON object ingestion here if needed
                st.write("JSON data loaded successfully (single object).")
                st.json(data)
                episode_name = f"JSON Ingest - {file_name}"
                episode_body = json.dumps(data) # Ensure it's a string
                source_type = EpisodeType.json
                source_description = "Uploaded JSON file"

                async def ingest_single_data():
                    try:
                        await graphiti.add_episode(
                            name=episode_name,
                            episode_body=episode_body,
                            source=source_type,
                            source_description=source_description,
                            valid_at=datetime.now(),
                        )
                        st.success("Single JSON object ingested into Graphiti successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during single ingestion: {e}")

                if st.button("Ingest Single Data into Graphiti"):
                    asyncio.run(ingest_single_data())


        elif file_type.startswith("image/"):
            # Process image file using EasyOCR
            st.write("Extracting text from image...")
            img_bytes = uploaded_file.getvalue()
            # Initialize reader here for thread safety in Streamlit if not already initialized globally
            # reader = easyocr.Reader(['en'])
            extracted_text = reader.readtext(img_bytes, detail=0) # detail=0 returns just the text
            episode_body = "\n".join(extracted_text) # Join lines of text

            episode_name = f"Image Ingest - {file_name}"
            source_type = EpisodeType.text
            source_description = "Uploaded image file (text extracted by EasyOCR)"

            st.write("Image file uploaded and text extracted.")
            st.image(uploaded_file)
            st.subheader("Extracted Text:")
            st.text(episode_body)

            # Ingest into Graphiti
            st.write("Attempting to ingest data...")

            async def ingest_image_data():
                try:
                    await graphiti.add_episode(
                        name=episode_name,
                        episode_body=episode_body,
                        source=source_type,
                        source_description=source_description,
                        reference_time=datetime.now(),
                    )
                    st.success("Image text ingested into Graphiti successfully!")
                except Exception as e:
                    st.error(f"An error occurred during image text ingestion: {e}")

            if st.button("Ingest Image Text into Graphiti"):
                asyncio.run(ingest_image_data())


        else:
            st.warning(f"Unsupported file type: {file_type}")
            episode_name = None # Don't ingest if unsupported
            source_type = None
            source_description = None
            episode_body = None


    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")


st.subheader("Search Graphiti")
search_query = st.text_input("Enter search query:")

async def perform_search(query):
    try:
        
        st.write(f"Performing hybrid search for query '{query}'...")
        results = await graphiti.search(query)
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = GraphitiSearchResult(
                uuid=result.uuid,
                fact=result.fact,
                source_node_uuid=result.source_node_uuid if hasattr(result, 'source_node_uuid') else None
            )
            
            # Add temporal information if available
            if hasattr(result, 'valid_at') and result.valid_at:
                formatted_result.valid_at = str(result.valid_at)
            if hasattr(result, 'invalid_at') and result.invalid_at:
                formatted_result.invalid_at = str(result.invalid_at)
            
            formatted_results.append(formatted_result)
    

        st.write("Search Results:")
        st.write(formatted_results)

        st.write("Getting final answer from Gemini using google-genai...")
        # Assuming results is a string or can be easily converted to a string for the prompt
        results_text = str(formatted_results)
        final_answer = generate(results_text) # Call the modified generate function

        st.write("Final Answer from Gemini:")
        st.write(final_answer)

    except Exception as e:
        st.error(f"An error occurred during search: {e}")


if st.button("Search"):
    if search_query:
        # Need to handle async in Streamlit button click
        asyncio.run(perform_search(search_query))
    else:
        st.warning("Please enter a search query.")
