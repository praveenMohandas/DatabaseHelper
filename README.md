# Dune Helper Pipeline

## Introduction

The Dune Helper  is a the backend pipeline of Dune helper bot,an advanced query processing application that leverages FastAPI, PostgreSQL, and OpenAI's API for natural language processing. The pipeline takes in user queries, determines their intent, retrieves relevant context from a PostgreSQL database (using pgvector for embedding similarity), and optionally performs database actions such as adding, replacing, or deleting content. Additionally, it manages conversation history to provide context-aware responses.

## Features

- **Advanced Query Processing:** Combines multiple LLM calls to refine queries, extract intent,perform database actions, and generate final responses.
- **Context Retrieval:** Uses pgvector similarity search to retrieve the most relevant records from the database.
- **Database Operations:** Supports "add", "replace", and "delete" actions on the stored content.
- **Conversation Management:** Maintains conversation history via PostgreSQL for context in subsequent queries.
- **OpenAI Integration:** Utilizes OpenAI's API to power natural language understanding and response generation.

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- **PostgreSQL** (ensure it is installed and running)
- **Required Python Packages:**  
  Install dependencies via pip (ideally using a virtual environment):
  ```bash
  pip install -r requirements.txt
   ```

### Configuration File:
Ensure you have a config.py file with the following (adjust values as needed):
Database credentials: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
OpenAI API key: OPENAI_API_KEY
Other settings: TABLE_NAME, EMBEDDING_DIM, JSONL_FILE, and prompts such as INTENT_PROMPT, SECOND_LLM_PROMPT, THIRD_LLM_PROMPT

### Database Setup
Create the PostgreSQL Database:

Before running the application, create your database. For example, log into PostgreSQL and execute:

 ```sql

CREATE DATABASE your_database_name;
 ```
Replace your_database_name with the desired name.

Set Up the Database Schema:

 The script ( database.py) handles the creation of the required table and pgvector extension. To run this setup script:

 ```bash
python database.py
 ```
This script will:

Enable the pgvector extension if it isn’t already installed.
Create the table (as specified by config.TABLE_NAME) with columns for id, content, url, and embedding.
Ingest data from a JSONL file  provided.
### Running the API
Start the FastAPI Application:

Use Uvicorn to run the API server:

 ```bash
uvicorn query_service:app --reload
 ```


Access the Endpoints:

The main query endpoint is available at:
http://localhost:8000/query
or go to http://127.0.0.1:8000/docs to access UI of FastAPI and then you can pass your query to the query endpoint.
Database operations are available under the /db prefix (e.g., /db/add, /db/replace, /db/delete).
## How the Pipeline Works
- **User Query Submission:
The query endpoint stores the user query and checks for similar previous queries via the conversation history.
If the  current query is similar to previous queries LLM1 and LLM2 calls are skipped and goes straight to LLM3 call
LLM #1 – Intent Extraction:
An initial call to OpenAI’s API determines the query intent, action, and may refine the query for better retrieval.

- **Context Retrieval:
If needed (based on the determined action), the application retrieves relevant rows from the database using vector similarity search.

- **LLM #2 – Processing:
Combines the intent, action, and retrieved context to decide if a database action is required and prepares new content if necessary.

- **Database Operations (Optional):
Depending on the action (add, replace, delete), the corresponding API endpoint is called to update the database.

- **LLM #3 – Final Response:
A final LLM call generates the user response, incorporating conversation history and any additional context.

- **Conversation History Management:
All user and AI messages are stored in PostgreSQL, maintaining context across interactions.

## Folder Structure

DuneHelperPipeline/

├── config.py             # Configuration file (database credentials, API keys, model settings, prompts)

├── query_service.py      # Main FastAPI application (query endpoint and pipeline orchestration)

├── retrieval.py          # Module for context retrieval using pgvector similarity search

├── db_command.py         # FastAPI routes for database operations (add, replace, delete)

├── conversation_hist.py  # Conversation management and history logging via PostgreSQL using Langchain

├── llm_calls.py          # Functions to interact with OpenAI's API (LLM calls)

├── database.py           # Database setup script (table creation, pgvector extension, data ingestion)

├── logs/                 # Directory for log files (e.g., query_service.log)

└── README.md             # Readme

## Additional Notes
Logging:
Detailed logs are written to both the console and logs/query_service.log for troubleshooting.

Conversation History:
Managed via the ConversationManager class using the langchain_postgres library.

Customization:
You can modify the LLM prompts, retrieval logic, and database operations according to your project needs.


