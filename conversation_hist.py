import psycopg
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import numpy as np
import config
import logging

class ConversationManager:
    """
    Manages storing and retrieving conversation messages
    via langchain_postgres' PostgresChatMessageHistory.
    """

    def __init__(self, connection_string: str, table_name: str, session_id: str, logger: logging.Logger = None):
        # Use the provided logger, or fallback to module logger.
        self.logger = logger or logging.getLogger(__name__)
        
        # 1) Open a synchronous psycopg connection
        self.conn = psycopg.connect(connection_string)

        # 2) Create the table schema (only needed once, but safe to call each time)
        PostgresChatMessageHistory.create_tables(self.conn, table_name)

        # 3) Instantiate the chat history object
        self.chat_history = PostgresChatMessageHistory(
            "conversation_history",
            session_id,
            sync_connection=self.conn
        )

    def add_user_message(self, message_text: str) -> None:
        """Add a user message to the conversation."""
        self.chat_history.add_messages([HumanMessage(content=message_text)])

    def add_ai_message(self, message_text: str) -> None:
        """Add an AI/assistant message to the conversation."""
        self.chat_history.add_messages([AIMessage(content=message_text)])

    def add_system_message(self, message_text: str) -> None:
        """Add a system message to the conversation (optional)."""
        self.chat_history.add_messages([SystemMessage(content=message_text)])

    def get_conversation_history(self):
        """
        Return the entire conversation as a list of LangChain
        message objects (SystemMessage, HumanMessage, AIMessage).
        """
        return self.chat_history.get_messages()

    def convert_langchain_messages_to_openai(self, messages):
        converted = []
        for msg in messages:
            if msg.type == "human":
                converted.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                converted.append({"role": "assistant", "content": msg.content})
            elif msg.type == "system":
                converted.append({"role": "system", "content": msg.content})
            else:
                # fallback
                converted.append({"role": "user", "content": msg.content})
        return converted

    def clear_session(self):
        """Clear the conversation (deletes all messages for this session_id)."""
        self.chat_history.clear()

    def has_relevant_previous_query(self, current_query: str, threshold: float = 0.70) -> bool:
        """
        Compare the current query to all previous user queries using sentence embeddings.
        Returns True if any previous query has a cosine similarity above the given threshold.
        
        Parameters:
            current_query (str): The new user query.
            threshold (float): The cosine similarity level required (between -1 and 1). Default is 0.92.
        
        Returns:
            bool: True if a similar previous query exists, False otherwise.
        """
        # Compute the embedding for the current query.
        current_embedding = config.embedding_model.encode(current_query)
        self.logger.debug("Current query: %s", current_query)
        
        previous_messages = self.get_conversation_history()
        for msg in previous_messages:
            if hasattr(msg, "type") and msg.type == "human":
                self.logger.debug("Comparing with previous query: %s", msg.content)
                previous_embedding = config.embedding_model.encode(msg.content)
                # Compute cosine similarity
                similarity = np.dot(current_embedding, previous_embedding) / (np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding))
                self.logger.debug("Similarity with [%s]: %.4f", msg.content, similarity)
                if similarity >= threshold:
                    self.logger.debug("Match found with similarity: %.4f", similarity)
                    return True
        self.logger.debug("No match found for query: %s", current_query)
        return False
