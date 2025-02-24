import openai
import json
import config
import logging

logger = logging.getLogger("llm_calls")
logger.setLevel(logging.DEBUG)

def call_openai(messages, temperature=0.0, max_tokens=500):
    """
    Generic function to call the OpenAI API.
    """
    openai.api_key = config.OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    result = resp.choices[0].message.content.strip()
    return result

def call_llm1(user_query: str) -> dict:
    """
    LLM #1: Determine intent, action, and optionally refine the query.
    Returns a parsed JSON object.
    """
    messages = [
        {"role": "system", "content": config.INTENT_PROMPT.strip()},
        {"role": "user", "content": user_query}
    ]
    llm1_raw = call_openai(messages, temperature=0.0, max_tokens=500)
    logger.debug("LLM #1 raw output: %s", llm1_raw)
    try:
        parsed = json.loads(llm1_raw)
    except Exception as e:
        raise Exception(f"LLM #1 invalid JSON: {e}")
    return parsed

def call_llm2(user_input: dict) -> dict:
    """
    LLM #2: Process the intent, action, and retrieved context.
    Returns a parsed JSON object.
    """
    messages = [
        {"role": "system", "content": config.SECOND_LLM_PROMPT.strip()},
        {"role": "user", "content": json.dumps(user_input)}
    ]
    llm2_raw = call_openai(messages, temperature=0.0, max_tokens=1500)
    logger.debug("LLM #2 raw output: %s", llm2_raw)
    try:
        parsed = json.loads(llm2_raw)
    except Exception as e:
        raise Exception(f"LLM #2 invalid JSON: {e}")
    return parsed

def call_llm3(user_query: str, conversation_manager, additional_context: dict = None) -> str:
    """
    LLM #3: Generate the final response by including conversation history.
    This function expects the conversation_manager to provide access to the chat history.
    
    Parameters:
        user_query: The current user query.
        conversation_manager: An instance of ConversationManager.
        additional_context: A dictionary containing additional details from previous pipeline steps.
        
    Returns:
        The raw output from LLM #3.
    """
    previous_messages = conversation_manager.get_conversation_history()
    previous_messages = conversation_manager.convert_langchain_messages_to_openai(previous_messages)
    chat_history_str = "Chat History:\n" + "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in previous_messages]
    )
    chat_history_message = {"role": "system", "content": chat_history_str}
    system_prompt = {"role": "system", "content": config.THIRD_LLM_PROMPT.strip()}
    
    # Merge the current query with any additional context
    payload = {"query": user_query}
    if additional_context:
        payload.update(additional_context)
    final_user_message = {"role": "user", "content": json.dumps(payload)}
    
    messages_for_llm3 = [system_prompt, final_user_message, chat_history_message]
    logger.debug("LLM #3 input: %s", messages_for_llm3)
    
    llm3_raw = call_openai(messages_for_llm3, temperature=0.8, max_tokens=200)
    logger.debug("LLM #3 raw output: %s", llm3_raw)
    return llm3_raw
