
import psycopg2
from psycopg2.extras import RealDictCursor
import config

def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )

def get_query_embedding(query: str):
    emb = config.embedding_model.encode(query)
    return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

def get_relevant_context(refined_query: str, top_n: int = 3):
    """
    Searches for the top relevant content from the database using pgvector similarity.
    Returns a list of dicts with keys {id, content, url}.
    """
    embedding = get_query_embedding(refined_query)
    embedding_str = f'[{",".join(map(str, embedding))}]'
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query_sql = f"""
    SELECT id, content, url
    FROM {config.TABLE_NAME}
    ORDER BY embedding <-> %s
    LIMIT %s;
    """
    try:
        cur.execute(query_sql, (embedding_str, top_n))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return rows  # list of dicts with {id, content, url}
