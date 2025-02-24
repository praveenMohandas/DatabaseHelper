import json
import os
import psycopg2
from psycopg2.extras import execute_values
import config

def get_connection():
    """
    Establish a connection to the PostgreSQL database using details from config.py.
    """
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )
    return conn

def setup_table():
    """
    Sets up the database table by ensuring the pgvector extension is enabled and
    creating the table with columns: id, content, url, and embedding.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Create the pgvector extension if it does not exist
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Create the table if it does not exist
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        content TEXT,
        url TEXT,
        embedding vector({config.EMBEDDING_DIM})
    );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table setup completed.")

def compute_embedding(content):
    """
    Compute embedding using the SentenceTransformer model from config.embedding_model.
    The model returns a numpy array that is converted to a list of floats.
    """
    embedding = config.embedding_model.encode(content)
    # Convert numpy array to list if needed
    return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

def ingest_jsonl(file_path=config.JSONL_FILE):
    """
    Reads a JSONL file line by line, computes embeddings for each content, and inserts
    the records into the database.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    conn = get_connection()
    cur = conn.cursor()
    data_to_insert = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            content = record.get("content", "").replace("\n", " ")
            url = record.get("url", "")
            embedding = compute_embedding(content)
            # Convert the embedding list into the PostgreSQL vector literal format (e.g., [0.1,0.2,...])
            embedding_str = f'[{",".join(map(str, embedding))}]'
            data_to_insert.append((content, url, embedding_str))
    
    # Bulk insert using execute_values for efficiency
    insert_query = f"INSERT INTO {config.TABLE_NAME} (content, url, embedding) VALUES %s"
    execute_values(cur, insert_query, data_to_insert)
    conn.commit()
    cur.close()
    conn.close()
    print(f"Ingested {len(data_to_insert)} records.")

if __name__ == '__main__':
    setup_table()
    ingest_jsonl()
