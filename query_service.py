import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
import logging
import uuid

import config
from retrieval import get_relevant_context
from db_command import router as db_command_router
from conversation_hist import ConversationManager  
from llm_calls import call_llm1, call_llm2, call_llm3

session_id = str(uuid.uuid4())

logger = logging.getLogger("query_service")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
fh = logging.FileHandler("logs/query_service.log")
logger.addHandler(ch)
logger.addHandler(fh)

app = FastAPI(title="Dune helper Pipeline")
app.include_router(db_command_router, prefix="/db")

# Instantiate your ConversationManager once
CONNECTION_STRING = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
conversation_manager = ConversationManager(
    connection_string=CONNECTION_STRING,
    table_name="conversation_history",
    session_id=str(uuid.uuid4()),
    logger=logger
)

class IntentResponse(BaseModel):
    intent: str
    action: Optional[str] = None
    old_feature: Optional[str] = None
    new_feature: Optional[str] = None
    refined_query: Optional[str] = None

class SecondLLMOutput(BaseModel):
    new_content: Optional[str] = None
    call_to_db: bool

class PipelineResponse(BaseModel):
    intent: str
    action: Optional[str] = None
    Changed_ids: Optional[List[Any]] = None
    retrieved_rows: List[Any]
    new_content: Optional[str] = None
    call_to_db: bool
    final_user_response: str

@app.post("/query", response_model=PipelineResponse)
def query_endpoint(user_query: str):
    # 1) Store user query

    logger.debug("Stored user query: %s", user_query)
    
    # Check if a similar query exists
    if conversation_manager.has_relevant_previous_query(user_query):
        logger.debug("Found relevant previous query. Skipping to LLM #3.")
        # Directly call LLM #3 if a similar query was found.
        llm3_raw = call_llm3(user_query, conversation_manager)
        logger.debug("LLM #3 raw output (from similar query branch): %s", llm3_raw)
        conversation_manager.add_ai_message(llm3_raw)
        return PipelineResponse(
            intent="",
            action=None,
            Changed_ids=[],
            retrieved_rows=[],
            new_content=None,
            call_to_db=False,
            final_user_response=llm3_raw
        )
    conversation_manager.add_user_message(user_query)
    # STEP 1: LLM #1 
    try:
        parsed1 = call_llm1(user_query)
        logger.debug("LLM #1 parsed output: %s", parsed1)
        # Validate LLM #1 response using Pydantic
        intent_data = IntentResponse.parse_obj(parsed1)
    except Exception as e:
        logger.error("LLM #1 response validation failed: %s", str(e))
        fallback_message = "I'm having trouble understanding your request. Could you please rephrase or provide more details?"
        conversation_manager.add_ai_message(fallback_message)
        return PipelineResponse(
            intent="fallback",
            action=None,
            Changed_ids=[],
            retrieved_rows=[],
            new_content=None,
            call_to_db=False,
            final_user_response=fallback_message
        )
    
    #  STEP 2: Retrieval 
    rows = []
    if intent_data.action and intent_data.action.lower() in ("retrieve", "replace", "delete"):
        if intent_data.refined_query:
            logger.debug("Using refined query for retrieval: %s", intent_data.refined_query)
            rows = get_relevant_context(intent_data.refined_query)
        else:
            logger.debug("Using original user query for retrieval: %s", user_query)
            rows = get_relevant_context(user_query)
        logger.debug("Retrieved rows: %s", rows)
    
    # STEP 3: LLM #2
    user_input_llm2 = {
        "intent": intent_data.intent,
        "action": intent_data.action,
        "old_feature": parsed1.get("old_feature"),
        "new_feature": parsed1.get("new_feature"),
        "retrieved_context": rows
    }
    logger.debug("LLM #2 input: %s", user_input_llm2)
    try:
        parsed2 = call_llm2(user_input_llm2)
        logger.debug("LLM #2 parsed output: %s", parsed2)
        # Validate LLM #2 response using Pydantic
        second_data = SecondLLMOutput.parse_obj(parsed2)
    except Exception as e:
        logger.error("LLM #2 response validation failed: %s", str(e))
        fallback_message = "I'm having trouble processing your request. Could you please rephrase or provide more details?"
        conversation_manager.add_ai_message(fallback_message)
        return PipelineResponse(
            intent=intent_data.intent,
            action=intent_data.action,
            Changed_ids=[],
            retrieved_rows=rows,
            new_content=None,
            call_to_db=False,
            final_user_response=fallback_message
        )
    
    # STEP 4: (Optional) Execute DB actions if needed
    changed_ids = []
    if second_data.call_to_db:
        logger.debug("DB action required. Execute DB actions here.")
        # Determine the DB action from the intent. Default to "add" if not specified.
        db_action = intent_data.action.lower() if intent_data.action else None
        logger.debug("Determined DB action: %s", db_action)

        if db_action == "add":
            payload = {"new_content": second_data.new_content}
            logger.debug("Sending DB add request with payload: %s", payload)
            response = requests.post("http://localhost:8000/db/add", json=payload)
            logger.debug("DB add response: %s", response.text)
            if response.ok:
                new_id = response.json().get("new_id")
                changed_ids.append(new_id)
            else:
                raise HTTPException(status_code=500, detail="DB add operation failed")
                
        elif db_action == "replace":
            # Assume that retrieved rows contain an 'id' field if available, otherwise treat the row as an ID.
            if rows and isinstance(rows, list) and len(rows) > 0:
                row_ids = []
                for row in rows:
                    if isinstance(row, dict) and "id" in row:
                        row_ids.append(row["id"])
                    else:
                        row_ids.append(row)
                logger.debug("Row IDs extracted for replacement: %s", row_ids)
            else:
                logger.error("No rows available for replacement.")
                raise HTTPException(status_code=500, detail="No rows available for replacement")
            payload = {"row_ids": row_ids, "new_content": second_data.new_content}
            logger.debug("Sending DB replace request with payload: %s", payload)
            response = requests.post("http://localhost:8000/db/replace", json=payload)
            logger.debug("DB replace response: %s", response.text)
            if response.ok:
                updated_id = response.json().get("updated_id")
                changed_ids.append(updated_id)
            else:
                raise HTTPException(status_code=500, detail="DB replace operation failed")
                
        elif db_action == "delete":
            if rows and isinstance(rows, list) and len(rows) > 0:
                row_ids = []
                for row in rows:
                    if isinstance(row, dict) and "id" in row:
                        row_ids.append(row["id"])
                    else:
                        row_ids.append(row)
                logger.debug("Row IDs extracted for deletion: %s", row_ids)
            else:
                logger.error("No rows available for deletion.")
                raise HTTPException(status_code=500, detail="No rows available for deletion")
            payload = {"row_ids": row_ids}
            logger.debug("Sending DB delete request with payload: %s", payload)
            response = requests.post("http://localhost:8000/db/delete", json=payload)
            logger.debug("DB delete response: %s", response.text)
            if response.ok:
                deleted_ids = response.json().get("deleted_ids", [])
                changed_ids.extend(deleted_ids)
            else:
                raise HTTPException(status_code=500, detail="DB delete operation failed")

    
    # STEP 5: LLM #3 
    third_input = {
        "intent": intent_data.intent,
        "action": intent_data.action,
        "row_ids": changed_ids,
        "context": rows,
        "new_content": second_data.new_content
    }
    logger.debug("LLM #3 additional context: %s", third_input)
    llm3_raw = call_llm3(user_query, conversation_manager, third_input)
    logger.debug("LLM #3 raw output: %s", llm3_raw)
    conversation_manager.add_ai_message(llm3_raw)
    #if you want to clear chats
    #conversation_manager.clear_session()
    return PipelineResponse(
        intent=intent_data.intent,
        action=intent_data.action,
        Changed_ids=changed_ids,
        retrieved_rows=rows,
        new_content=second_data.new_content,
        call_to_db=second_data.call_to_db,
        final_user_response=llm3_raw
    )
