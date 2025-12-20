# Ember AI — Full-Stack Conversational AI Platform

Ember AI is a full-stack conversational AI application designed and built as a production-oriented system, not a prototype.
It is developed in a way that it can be easily modified to fit any domain. This project combines both frontend and backend development, incorporating strong AI
models using groq inference and advanced libraries and frameworks like LangChain/LangGraph backed by Supabase database.
It integrates real-time LLM streaming, persistent memory, Retrieval-Augmented Generation (RAG), voice input, and secure authentication into a cohesive architecture.

The project emphasizes clear separation of concerns, backend-driven intelligence, and a thin, deterministic frontend.



## System Overview

Ember AI consists of two primary layers:

- **Backend**: Responsible for intelligence, memory, retrieval, orchestration, and persistence  
- **Frontend**: A lightweight web client responsible for interaction, streaming display, and user experience

All conversational logic, memory, and reasoning reside exclusively on the backend.



The frontend acts as a **transparent interface** to backend capabilities, avoiding duplicated logic or hidden state.



## Backend

### Backend Responsibilities

The backend is responsible for:

- User authentication and authorization
- Chat session management
- Streaming LLM responses
- Persistent conversation memory
- Retrieval-Augmented Generation (RAG)
- Audio transcription (ASR)
- Secure multi-chat isolation
- Backend-controlled system prompts and behavior

The backend is the **single source of truth** for all application state.



### Technology Stack (Backend)

- **Python**
- **FastAPI** — async API framework
- **LangGraph** — conversational state orchestration
- **LLM API** — streamed response generation
- **Vector Store** — document embeddings for RAG
- **Supabase** — authentication and user management
- **SQL Database** — chat and message persistence



### Chat & Memory Model

- Each user may have multiple independent chats
- Each chat has a unique `chat_id`
- Message history is persisted server-side
- Memory is reconstructed from stored messages on each request
- No client-side memory is trusted or reused

This guarantees deterministic behavior even across reloads or devices.



### Streaming Response Design

LLM responses are streamed token-by-token from the backend:

- Backend emits a streaming HTTP response
- Tokens are flushed as soon as they are generated
- The backend does not buffer or post-process full responses
- Streaming preserves responsiveness for long outputs

This design improves perceived performance and user experience.



### Retrieval-Augmented Generation (RAG)

Document ingestion follows a controlled backend pipeline:

1. Document upload
2. Text extraction
3. Chunking
4. Embedding generation
5. Vector storage
6. Contextual retrieval at query time

Documents are scoped per chat, preventing cross-conversation leakage.



### Voice Input (ASR)

- Audio is received as raw input
- Transcription occurs server-side
- Resulting text enters the same chat pipeline as typed input
- No special-case logic exists for voice vs text



### Security & Authentication

- JWT-based authentication via Supabase
- Chat ownership enforced server-side
- Unauthorized access returns immediate rejection
- Backend never trusts client-supplied state beyond identifiers



## Frontend

### Frontend Overview

The frontend is a **framework-free, vanilla JavaScript web client** designed for:

- Real-time streaming display
- Low latency
- Explicit control over request flow
- Minimal abstraction between UI and backend

The frontend intentionally avoids complex state management frameworks.



### Frontend Responsibilities

The frontend handles:

- Login / signup / logout
- Token lifecycle management
- Chat selection and creation
- Rendering streamed responses
- Markdown rendering
- File uploads for RAG
- Voice recording via browser APIs
- UI state synchronization with backend data

It does **not** perform reasoning, memory handling, or prompt logic.



### Technology Stack (Frontend)

- **HTML**
- **CSS**
- **Vanilla JavaScript**
- **Browser Streaming APIs**
- **MediaRecorder API** (voice input)
- **Markdown Renderer**



### Authentication Flow

- Access tokens are attached to all protected requests
- Refresh tokens are used for silent renewal
- Automatic refresh occurs on expiration
- On failure, the user is cleanly logged out

This ensures session continuity without compromising security.



### Chat Lifecycle

- Chats are fetched from the backend after authentication
- Selecting a chat loads its persisted message history
- New chats are created explicitly
- Deleting a chat removes all backend state
- Chat titles are generated and updated server-side

The frontend never reconstructs chat state locally.



### Streaming Handling

- Responses are consumed using `ReadableStream`
- Tokens are appended incrementally to the UI
- Markdown is re-rendered continuously
- Scroll position is automatically maintained

This enables a smooth, real-time conversational experience.



### UI Design Philosophy

The UI follows these principles:

- Thin client
- Backend-driven truth
- Explicit state transitions
- No speculative logic
- Predictable behavior

The frontend exists to **reflect**, not reinterpret, backend behavior.



## Deployment

- Backend runs as an async FastAPI service
- Frontend can be hosted statically (e.g. GitHub Pages)
- Communication occurs exclusively over HTTPS
- CORS is explicitly configured on the backend



## Why This Project Matters

From a systems engineering perspective, Ember AI demonstrates:

- Real-time LLM streaming integration
- Stateful conversational orchestration
- Secure multi-user chat isolation
- Practical RAG implementation
- Clean frontend–backend separation
- Production-oriented architectural thinking

This project prioritizes correctness, clarity, and maintainability over convenience abstractions.

---
