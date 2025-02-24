

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import psycopg2
import config
from typing import Optional


router = APIRouter()

def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD
    )

# 1) ADD
class AddRequest(BaseModel):
    new_content: Optional[str] = None

@router.post("/add")
def add_content(body: AddRequest):
    """
    Insert a new row with new_content.
    (In production you'd also compute embedding, etc.)
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO {config.TABLE_NAME} (content) VALUES (%s) RETURNING id;",
            (body.new_content,)
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "new_id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2) REPLACE
class ReplaceRequest(BaseModel):
    row_ids: list[int]
    new_content: Optional[str] = None

@router.post("/replace")
def replace_content(body: ReplaceRequest):
    """
    For each row_id, set content = new_content.
    """
    if not body.row_ids:
        return {"status": "no_rows", "message": "No row_ids provided"}

    try:
        conn = get_connection()
        cur = conn.cursor()
        # Only update the first row_id
        row_id = body.row_ids[0]
        cur.execute(
            f"UPDATE {config.TABLE_NAME} SET content = %s WHERE id = %s",
            (body.new_content, row_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "updated_id": row_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3) DELETE
class DeleteRequest(BaseModel):
    row_ids: list[int]

@router.post("/delete")
def delete_content(body: DeleteRequest):
    """
    Delete rows by ID.
    """
    if not body.row_ids:
        return {"status": "no_rows", "message": "No row_ids provided"}

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {config.TABLE_NAME} WHERE id = ANY(%s)",
            (body.row_ids,)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "deleted_ids": body.row_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
