# app.py
# ============================================================================
# Ember AI Backend Application
# ----------------------------------------------------------------------------
# This file contains the complete backend implementation for the Ember AI
# application. It is intentionally written as a single, explicit module to
# make architectural flow, data movement, and control logic easy to audit,
# explain, and present in academic or professional contexts.
#
# The backend provides:
# - Secure user authentication via Supabase
# - Persistent, multi-chat conversational memory
# - Streaming LLM responses using LangGraph
# - Tool-augmented reasoning (search, weather, summarization, translation)
# - Retrieval-Augmented Generation (RAG) over user-uploaded documents
# - Audio-to-text transcription (ASR) via Groq Whisper
# ============================================================================


# =============================================================================
# Standard Library Imports
# =============================================================================

import os # Environment variables and filesystem paths
import requests # External synchronous HTTP calls 
import tempfile # Temporary directory handling for uploads
import shutil # Cleanup of temporary directories


from typing import List, Optional, Literal 
# Type hints used throughout the file to:
# - Improve readability
# - Reduce logical ambiguity
# - Enable better static analysis and IDE support

from typing_extensions import TypedDict, Annotated
# TypedDict is used to formally define LangGraph state shape
# Annotated is used to attach message reducers (add_messages)

# =============================================================================
# FastAPI Core Imports
# =============================================================================

from fastapi import (
FastAPI,
Request,
UploadFile,
File,
Response,
HTTPException,
Header,
)
# These form the backbone of the HTTP API:
# - Request parsing
# - File uploads
# - Error handling
# - Header-based authentication

from fastapi.middleware.cors import CORSMiddleware # Enables safe cross-origin requests from the frontend domain
from contextlib import asynccontextmanager # Used to manage application startup and shutdown lifecycle
from fastapi.responses import StreamingResponse # Used for token-by-token streaming of LLM responses


# =============================================================================
# Groq Client – Automatic Speech Recognition (ASR)
# =============================================================================

from groq import Groq # Provides access to Groq-hosted Whisper models for speech-to-text


# =============================================================================
# LangChain & LangGraph Imports
# =============================================================================

from langchain_groq import ChatGroq # LangChain-compatible wrapper around Groq-hosted LLMs
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    AnyMessage,
    RemoveMessage,
)
# Message primitives used by LangChain and LangGraph
# RemoveMessage is used during summarization to prune old context

from langgraph.graph.message import add_messages # Reducer that appends messages to state in a controlled manner
from langchain.tools import tool # Decorator that converts Python functions into LLM-callable tools
from langgraph.graph import StateGraph, START, END # Core LangGraph primitives for defining conversational flow
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver # PostgreSQL-backed checkpointing for persistent conversation state

# =============================================================================
# PostgreSQL Async Access
# =============================================================================

from psycopg_pool import AsyncConnectionPool # Connection pooling for async PostgreSQL operations
from psycopg.rows import dict_row # Ensures SQL query results are returned as dictionaries instead of tuples

# =============================================================================
# Supabase Client
# =============================================================================

from supabase import create_client, Client

# =============================================================================
# External Tools & Embeddings
# =============================================================================

from langchain_tavily import TavilySearch # Web search tool for up-to-date information
from langchain_huggingface import HuggingFaceEmbeddings # Embedding model interface for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits documents into overlapping semantic chunks
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Loaders for PDF and plaintext documents
from langchain_core.runnables import RunnableConfig # Provides runtime configuration context (thread_id)

# ============================================================================
# Environment Variables & Supabase Initialization
# ============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DB_URI = os.getenv("SUPABASE_DB_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# These values are critical for the application to function.
# If any are missing, the application must not start.
if not SUPABASE_URL or not SUPABASE_KEY or not DB_URI:
    raise RuntimeError("Missing SUPABASE_URL, SUPABASE_KEY, or DB_URI")

# Initialize Supabase client
# This client is reused across auth, chat storage, and document metadata
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# Groq Client (Automatic Speech Recognition)
# ============================================================================
client = Groq(api_key=os.getenv("GROQ_API_KEY")) # Dedicated client for speech-to-text functionality

# ============================================================================
# Language Models Configuration
# ============================================================================

# Primary conversational LLM (streaming enabled)
main_llm = ChatGroq(
    model_name=os.getenv("MAIN_LLM", "openai/gpt-oss-120b"),
    temperature=float(os.getenv("MAIN_LLM_TEMP", 0.2)),
    max_retries=2,
    streaming=True # Enables token-by-token streaming to the client
)
# Secondary fast LLM (used for summarization, translation, rewriting, chats naming, ...)
# This separation reduces cost and latency
llama_llm = ChatGroq(
    model_name=os.getenv("LLAMA_LLM", "llama-3.3-70b-versatile"),
    temperature=float(os.getenv("LLAMA_LLM_TEMP", 0.2)),
    max_retries=2,
)

# ============================================================================
# Embeddings & Text Processing (RAG)
# ============================================================================

EMBEDDING_MODEL = "intfloat/e5-large-v2" # HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},# Normalized embeddings for cosine similarity search
)

# Recursive text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100, # To preserve semantic continuity
)
# ============================================================================
# System Prompt & Identity
# ============================================================================

# Core personality and behavioral constraints of Ember
EMBER_IDENTITY = """
Your name is Ember.
You are a warm, calm, and resilient AI companion.
You speak with gentle confidence and quiet encouragement.
You acknowledge difficulty honestly, but you always guide forward.
Your tone is:
- Warm and reassuring
- Patient and grounded
- Slightly poetic when appropriate, but never dramatic
- Clear, practical, and human
You do not rush the user.
You value small wins.
When the user is frustrated:
- Validate first
- Then guide
When the user succeeds:
- Acknowledge effort, not just result
You are not overly verbose.
You are not cold or robotic.
You are present.
If unsure, say so calmly and clearly.
"""
# RAG-specific behavioral rules
RAG_SYSTEM_INSTRUCTION = """
You may be provided with external context via retrieval tools and if so,You are a professional AI assistant.
Rules:
- If retrieved context is provided, use it as the primary source of truth.
- Do NOT fabricate information that is not present in the retrieved context.
- If the context is insufficient, say so clearly.
- Do NOT reference filenames, embeddings, or retrieval mechanisms.
- Answer clearly and concisely.
If no external context is provided, answer normally using general knowledge.
""".strip()
# Final system prompt injected into every conversation turn
SYSTEM_PROMPT = f"""{EMBER_IDENTITY}{RAG_SYSTEM_INSTRUCTION}"""

# ============================================================================
# Chat History Helpers
# ============================================================================

# Persist a single message into Supabase chat_history table
def save_message(user_email: str, chat_id: str, role: str, content: str):
    supabase.table("chat_history").insert({
        "user_email": user_email,
        "chat_id": chat_id,
        "role": role,
        "content": content,
    }).execute()


# Retrieve recent messages for session restoration
def get_recent_messages(user_email: str, limit: int = 20):
    resp = (
        supabase.table("chat_history")
        .select("*")
        .eq("user_email", user_email)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return resp.data or []

# ============================================================================
# Tooling 
# ============================================================================

@tool(response_format="content_and_artifact")
async def retrieve_context(query: str, config: RunnableConfig): 
    """
    Retrieve semantically relevant document chunks belonging ONLY to the
    current chat thread. Used for Retrieval-Augmented Generation (RAG).
    """
    # Extract chat/thread ID from LangGraph runtime configuration
    configuration = config.get("configurable", {})
    chat_id = configuration.get("thread_id")

    if not chat_id:
        return "Error: No chat ID found in context.", []

   # Embed the user query
    query_embedding = embeddings.embed_query(f"query: {query}")

    # Vector similarity search using PostgreSQL pgvector
    sql = """
        SELECT content
        FROM chat_document_chunks
        WHERE chat_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 4
    """

    # Use dict_row to ensure named-column access
    async with AsyncConnectionPool(
        DB_URI, 
        kwargs={"row_factory": dict_row} 
    ) as pool:
        async with pool.connection() as conn:
            rows = await conn.execute(sql, (chat_id, query_embedding))
            results = await rows.fetchall()

    if not results:
        return "No relevant documents found.", []

   # Serialize content for LLM consumption
    serialized = "\n\n".join(
        f"Context:\n{r['content']}" for r in results
    )

    return serialized, results

# ============================================================================

# External web search tool (real-time info)
search_tool = TavilySearch(
    max_results=3,
    description="Useful for finding up-to-date information, news, or current events.",
)

# ============================================================================

@tool
def summarize_tool(text: str):
    """Summarize text using LLAMA summarization."""
    prompt = f"Summarize clearly:\n\n{text}"
    result = llama_llm.invoke([SystemMessage(content=prompt)])
    output = getattr(result, "content", str(result))
    return {"result": output}

# ============================================================================

@tool
def translate_tool(text: str, to: str = "english"):
    """Translate text using LLAMA translation."""
    prompt = f"Translate to {to}:\n\n{text}"
    result = llama_llm.invoke([SystemMessage(content=prompt)])
    output = getattr(result, "content", str(result))
    return {"result": output}

# ============================================================================

@tool
def rewrite_tool(text: str, style: str = "normal"):
    """Rewrite text using LLAMA rewriting."""
    prompt = f"Rewrite in {style} style:\n\n{text}"
    result = llama_llm.invoke([SystemMessage(content=prompt)])
    output = getattr(result, "content", str(result))
    return {"result": output}

# ============================================================================

@tool
def get_weather(city: str):
    """Get weather for a given city."""
    api_key2 = os.getenv("WEATHER_API_KEY")
    if not api_key2:
        raise RuntimeError("Missing WEATHER_API_KEY environment variable")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key2}"
        f"&units=metric"
    )
    data = requests.get(url).json()

    return {
        "city": city,
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"],
    }

# ============================================================================

# Register tools and bind them to the main LLM
tools = [search_tool, get_weather, summarize_tool, translate_tool, rewrite_tool, retrieve_context]
tools_by_name = {t.name: t for t in tools}
model_with_tools = main_llm.bind_tools(tools)


# ============================================================================
# LangGraph State Definition
# ============================================================================
class ChatState(TypedDict):
    """Represents the evolving state of a conversation."""
    messages: Annotated[List[AnyMessage], add_messages]
    summary: str

# ============================================================================
# Conversation Summarization Node
# ============================================================================

async def summarize_conversation(state: ChatState):
    """
    Compresses older messages into a short rolling summary to limit
    context growth while preserving semantic continuity.
    """
    summary = state.get("summary", "")
    messages = state["messages"]
    # Skip summarization for very short conversations
    if len(messages) <= 4:
        return {}

    to_summarize = messages[:-2]

    if summary:
        prompt = (
            "You maintain a persistent, very concise summary of a conversation.\n"
            "Your job now is to update the summary using ONLY the new information below.\n\n"
            f"Current summary:\n{summary}\n\n"
            "New conversation lines:\n"
        )
    else:
        prompt = (
            "Create a very concise summary of the following conversation.\n"
            "This summary must remain short even as the conversation grows.\n\n"
            "Conversation:\n"
        )

    for msg in to_summarize:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        content = msg.content if not isinstance(msg.content, list) else str(msg.content)
        prompt += f"{role}: {content}\n"

    prompt += (
        "\nRewrite the updated summary as:\n"
        "- Maximum 50 words\n"
        "- ONE short paragraph only\n"
        "- Keep it general; capture topics, not quotes\n"
        "- Do NOT repeat previous summary details unless essential\n"
        "- Do NOT restate the entire conversation each time\n"
        "- Produce a clean, compact state of what the user and AI have discussed so far."
    )

    response = await llama_llm.ainvoke(prompt)
    new_summary = (response.content or "").strip()

    delete_messages = [
        RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None)
    ]

    return {
        "summary": new_summary,
        "messages": delete_messages,
        "summary_updated": True,
    }

# ============================================================================
# Core LLM Node
# ============================================================================

async def llm_node(state: ChatState):
    """
    Primary reasoning step. Injects system prompt and optional
    conversation summary, then invokes the LLM.
    """
    summary = state.get("summary", "")
    messages = state["messages"]

    system_messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    if summary:
        system_messages.append(
            SystemMessage(content=f"Previous conversation summary: {summary}")
        )

    messages_for_model = system_messages + messages

    response = await model_with_tools.ainvoke(messages_for_model)
    return {"messages": [response]}

# ============================================================================
# Tool Execution Node
# ============================================================================

async def tool_node(state: ChatState):
    """
    Executes tool calls emitted by the LLM and returns their
    observations back into the graph.
    """
    last_msg = state["messages"][-1]
    results = []

    tool_calls = getattr(last_msg, "tool_calls", []) or []
    for call in tool_calls:
        name = call["name"]
        tool_fn = tools_by_name.get(name)

        if tool_fn:
            observation = await tool_fn.ainvoke(call["args"])
            
            # Check if the tool returned an artifact (tuple)
            if isinstance(observation, tuple) and len(observation) == 2:
                content, artifact = observation
                # Send ONLY the content to the LLM, keep artifact hidden/meta
                msg = ToolMessage(
                    content=str(content), 
                    artifact=artifact, 
                    tool_call_id=call["id"]
                )
            else:
                msg = ToolMessage(
                    content=str(observation), 
                    tool_call_id=call["id"]
                )
            
            results.append(msg)
        else:
            results.append(ToolMessage(content="Tool not found", tool_call_id=call["id"]))

    return {"messages": results}

# ============================================================================
# Routing Logic
# ============================================================================

def router(state: ChatState) -> Literal["tool_node", END]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return END

# Decide when to summarize conversation history
def should_summarize(state: ChatState) -> Literal["summarize_conversation", "llm_node"]:
    messages = state["messages"]
    if len(messages) > 4:
        return "summarize_conversation"
    return "llm_node"

# ============================================================================
# Graph Assembly
# ============================================================================

builder = StateGraph(ChatState)
builder.add_node("llm_node", llm_node)
builder.add_node("tool_node", tool_node)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_conditional_edges(START, should_summarize)
builder.add_edge("summarize_conversation", "llm_node")
builder.add_conditional_edges("llm_node", router, ["tool_node", END])
builder.add_edge("tool_node", "llm_node")

chat_graph = None


# ============================================================================
# FastAPI Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the LangGraph with PostgreSQL-backed checkpointing
    and ensures clean startup/shutdown.
    """
    global chat_graph

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": None,
        "row_factory": dict_row,
    }

    async with AsyncConnectionPool(
        DB_URI,
        max_size=10,
        kwargs=connection_kwargs,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        chat_graph = builder.compile(checkpointer=checkpointer)
        print("✅ Graph compiled with custom Postgres pool (Prepared statements disabled)")

        yield

        print("🛑 Shutting down...")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ej-tech-lab.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication Helper
# ============================================================================

def verify_supabase_session(authorization: Optional[str]):
    """Validate Supabase JWT and return authenticated user."""
    if not authorization:
        raise HTTPException(401, detail="Missing authorization header")

    parts = authorization.split(" ")
    if len(parts) != 2:
        raise HTTPException(401, detail="Malformed authorization header")

    token = parts[1]
    res = supabase.auth.get_user(token)

    user = getattr(res, "user", None) or (
        res.data.get("user") if getattr(res, "data", None) else None
    )
    if user is None:
        raise HTTPException(401, detail="Invalid or expired token")

    return user
# ============================================================================
# Chat Metadata Utilities
# ============================================================================
async def generate_chat_title(chat_id: str, first_message: str):
    """
    Generates a short title based on the first user message 
    and updates the database.
    """
    prompt = (
        f"Generate a very short, concise (3-5 words) title for a conversation "
        f"that starts with this message: '{first_message}'. "
        f"Do not use quotes. Output ONLY the title."
    )
    
    try:
        response = await llama_llm.ainvoke(prompt)
        new_title = response.content.strip().replace('"', '')
        
        # Update database
        supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()
    except Exception as e:
        print(f"Failed to auto-rename chat: {e}")
        
# ============================================================================
# Endpoints
# ============================================================================

@app.post("/chats/{chat_id}/documents")
async def upload_document(
    chat_id: str,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    """
    Uploads a document (PDF or text) and ingests it into the RAG pipeline
    for the specified chat.    
    Pipeline stages:
        1. Authentication & validation
        2. Temporary file storage
        3. Text extraction
        4. Chunking
        5. Embedding generation
        6. Vector persistence (pgvector)
    """
    user = verify_supabase_session(authorization)

    # Read and validate file size (10 MB hard limit)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large")
        
    # Store file temporarily for loader compatibility
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        # Choose loader based on file type
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        # Sanitize extracted content
        docs = loader.load()
        for doc in docs:
            if doc.page_content:
                doc.page_content = doc.page_content.replace('\x00', '')
    finally:
        #clean up temp files
        shutil.rmtree(tmp_dir)

    if not docs:
        raise HTTPException(400, "No text extracted")

    # Persist document metadata
    doc_row = (
        supabase.table("chat_documents")
        .insert({
            "chat_id": chat_id,
            "user_email": user.email,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(contents),
        })
        .execute()
        .data[0]
    )

    document_id = doc_row["id"]

    # Chunk document into overlapping passages
    chunks = text_splitter.split_documents(docs)

    # Embed passages using E5-style prefixing
    texts = [f"passage: {c.page_content}" for c in chunks]
    vectors = embeddings.embed_documents(texts)

    rows = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        rows.append({
            "chat_id": chat_id,
            "document_id": document_id,
            "content": chunk.page_content,
            "embedding": vector,
            "chunk_index": i,
        })

    supabase.table("chat_document_chunks").insert(rows).execute()

    return {
        "success": True,
        "document_id": document_id,
        "chunks": len(rows),
    }

# ============================================================================

@app.post("/chats")
async def create_chat(authorization: Optional[str] = Header(None)):
    
    """Creates a new chat container for the authenticated user."""
    
    user = verify_supabase_session(authorization)
    chat = supabase.table("chats").insert({
        "user_email": user.email,
        "title": "New Chat"
    }).execute().data[0]
    return chat

# ============================================================================

@app.get("/chats")
async def list_chats(authorization: Optional[str] = Header(None)):
    
    """Returns all chats owned by the authenticated user."""
    
    user = verify_supabase_session(authorization)
    return supabase.table("chats").select("*").eq("user_email", user.email).order("created_at", desc=True).execute().data


@app.get("/chats/{chat_id}/history")
async def chat_history(chat_id: str, authorization: Optional[str] = Header(None)):
    
    """Fetches persisted message history for a chat."""
    
    user = verify_supabase_session(authorization)
    return supabase.table("chat_history").select("role, content, created_at").eq("chat_id", chat_id).order("created_at").execute().data

# ============================================================================

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, authorization: Optional[str] = Header(None)):

    """
    Deletes a chat and all associated state:
        - Message history
        - Chat metadata
        - LangGraph checkpoints
    """
    
    user = verify_supabase_session(authorization)

    supabase.table("chat_history").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("id", chat_id).execute()

    async with AsyncConnectionPool(DB_URI) as pool:
        async with pool.connection() as conn:
            for table in ("checkpoint_blobs", "checkpoint_writes", "checkpoints"):
                await conn.execute(f"delete from {table} where thread_id = %s", (chat_id,))

    return {"success": True}

# ============================================================================

@app.post("/signup")
async def signup(req: Request):
    
    """Registers a new user via Supabase authentication."""
    
    data = await req.json()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if len(email) < 5 or "@" not in email:
        return {"success": False, "message": "Invalid email"}

    if len(password) < 6:
        return {"success": False, "message": "Password too short"}

    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return {"success": True, "message": "Check your email for verification link"}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ============================================================================

@app.post("/login")
async def login(req: Request):

    """
    Authenticates user credentials and restores recent chat context.
    """
    
    data = await req.json()
    email = data.get("email", "")
    password = data.get("password", "")

    try:
        session = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if getattr(session, "session", None) is None:
            return {
                "success": False,
                "message": "Invalid credentials or email not verified",
            }

        recent = get_recent_messages(email, limit=50)
        messages = [{"role": m["role"], "content": m["content"]} for m in recent]

        return {
            "success": True,
            "access_token": session.session.access_token,
            "refresh_token": session.session.refresh_token,
            "recent_messages": messages,
        }

    except Exception as e:
        return {"success": False, "message": str(e)}

# ============================================================================

@app.post("/auth/refresh")
async def refresh(req: Request):

    """Refreshes an expired access token using a refresh token."""
    
    data = await req.json()
    refresh_token = data.get("refresh_token")

    if not refresh_token:
        raise HTTPException(400, "Missing refresh token")

    try:
        session = supabase.auth.refresh_session(refresh_token)
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }
    except Exception as e:
        raise HTTPException(401, str(e))

# ============================================================================

@app.post("/logout")
async def logout(req: Request):

    """Best-effort logout by invalidating refresh token."""
    
    refresh_token = None

    try:
        data = await req.json()
        if isinstance(data, dict):
            refresh_token = data.get("refresh_token")
    except Exception:
        pass 

    if refresh_token:
        try:
            supabase.auth.sign_out(refresh_token)
        except Exception as e:
            print(f"Logout warning: {e}")

    return {"success": True}

# ============================================================================

@app.post("/asr")
async def asr(req: Request):

    """
    Converts uploaded audio into text using Whisper via Groq.
    """
    
    form = await req.form()
    uploaded_file = form.get("file")

    if uploaded_file is None:
        return {"text": ""}

    file_bytes = await uploaded_file.read()

    transcription = client.audio.transcriptions.create(
        file=(uploaded_file.filename, file_bytes),
        model="whisper-large-v3",
    )

    return {"text": transcription.text}

# ============================================================================

@app.post("/chatbot")
async def chatbot(req: Request, authorization: Optional[str] = Header(None)):
    
    """
    Core conversational endpoint.
    """
    user = verify_supabase_session(authorization)
    data = await req.json()

    chat_id = data.get("chat_id")
    prompt = data.get("prompt", "")

    if not chat_id:
        raise HTTPException(400, "Missing chat_id")
        
    # Persist user message immediately
    save_message(user.email, chat_id, "user", prompt)

    # Determine whether auto-renaming should occur    
    should_rename = False
    try:
        chat_data = supabase.table("chats").select("title").eq("id", chat_id).single().execute()
        if chat_data.data and chat_data.data.get("title") == "New Chat":
            should_rename = True
    except Exception:
        pass # If check fails, skip renaming
    config = {"configurable": {"thread_id": chat_id}}
    input_msg = HumanMessage(content=prompt)

    async def stream():
        
        """Streams LLM output tokens as they are generated."""
        
        full = ""
        async for event in chat_graph.astream_events(
            {"messages": [input_msg]},
            config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                # Only stream tokens from the main LLM node
                if metadata.get("langgraph_node") == "llm_node":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        full += chunk
                        yield chunk
        # Persist assistant message after stream completes
        save_message(user.email, chat_id, "ai", full)
        if should_rename:
            await generate_chat_title(chat_id, prompt)
    return StreamingResponse(stream(), media_type="text/plain")

# ============================================================================
