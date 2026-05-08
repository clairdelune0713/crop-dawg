import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = os.getenv("DATABASE_URL")

# Colors in order: red, blue, yellow, green, orange
COLOR_PALETTE = [
    {"name": "red", "hex": "#FF0000", "bgr": (0, 0, 255)},
    {"name": "blue", "hex": "#0000FF", "bgr": (255, 0, 0)},
    {"name": "yellow", "hex": "#FFFF00", "bgr": (0, 255, 255)},
    {"name": "green", "hex": "#00FF00", "bgr": (0, 255, 0)},
    {"name": "orange", "hex": "#FFA500", "bgr": (0, 165, 255)},
]

def get_db_connection():
    if all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS character_colors (
                id SERIAL PRIMARY KEY,
                user_email TEXT NOT NULL,
                project_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                color_name TEXT NOT NULL,
                color_hex TEXT NOT NULL,
                color_bgr TEXT NOT NULL,
                embedding FLOAT[],
                storyboard_number INTEGER,
                grid_number INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_email, project_id, character_name)
            );
        """)
        # Add new columns if they don't exist
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='embedding') THEN
                    ALTER TABLE character_colors ADD COLUMN embedding FLOAT[];
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='storyboard_number') THEN
                    ALTER TABLE character_colors ADD COLUMN storyboard_number INTEGER;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='grid_number') THEN
                    ALTER TABLE character_colors ADD COLUMN grid_number INTEGER;
                END IF;
            END $$;
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()

def record_character_color(user_email, project_id, character_name, embedding=None, storyboard_number=None, grid_number=None):
    """
    Records the character color mapping. Assigns the next available color from the palette.
    If the character already has a color for this project, it returns that mapping.
    """
    if not user_email or not project_id:
        return None

    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if already exists
        cur.execute("""
            SELECT color_name, color_hex, color_bgr FROM character_colors 
            WHERE user_email = %s AND project_id = %s AND character_name = %s
        """, (user_email, project_id, character_name))
        existing = cur.fetchone()
        if existing:
            return existing

        # Get count of characters for this project to determine next color
        cur.execute("""
            SELECT COUNT(*) FROM character_colors 
            WHERE user_email = %s AND project_id = %s
        """, (user_email, project_id))
        count = cur.fetchone()['count']
        
        color_idx = count % len(COLOR_PALETTE)
        color = COLOR_PALETTE[color_idx]
        
        bgr_str = f"({color['bgr'][0]},{color['bgr'][1]},{color['bgr'][2]})"
        
        cur.execute("""
            INSERT INTO character_colors (user_email, project_id, character_name, color_name, color_hex, color_bgr, embedding, storyboard_number, grid_number)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING color_name, color_hex, color_bgr
        """, (
            user_email, project_id, character_name, 
            color['name'], color['hex'], bgr_str, 
            embedding.tolist() if embedding is not None else None,
            storyboard_number, grid_number
        ))
        
        result = cur.fetchone()
        conn.commit()
        return result
    finally:
        cur.close()
        conn.close()

def get_project_characters(user_email, project_id, storyboard_number=None, grid_number=None):
    """Retrieves all registered characters for a specific project, optionally filtered by storyboard/grid."""
    if not user_email or not project_id:
        return []

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = """
            SELECT character_name, color_name, color_hex, color_bgr, embedding 
            FROM character_colors 
            WHERE user_email = %s AND project_id = %s
        """
        params = [user_email, project_id]
        
        if storyboard_number is not None:
            query += " AND storyboard_number = %s"
            params.append(storyboard_number)
        if grid_number is not None:
            query += " AND grid_number = %s"
            params.append(grid_number)
            
        cur.execute(query, tuple(params))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
