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
                nx1 INTEGER,
                ny1 INTEGER,
                nx2 INTEGER,
                ny2 INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Drop the old constraint and add the new one
        cur.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'character_colors_user_email_project_id_character_name_key'
                ) THEN
                    ALTER TABLE character_colors DROP CONSTRAINT character_colors_user_email_project_id_character_name_key;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'character_colors_unique_grid'
                ) THEN
                    ALTER TABLE character_colors ADD CONSTRAINT character_colors_unique_grid UNIQUE(user_email, project_id, character_name, storyboard_number, grid_number);
                END IF;
            END $$;
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
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='nx1') THEN
                    ALTER TABLE character_colors ADD COLUMN nx1 INTEGER;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='ny1') THEN
                    ALTER TABLE character_colors ADD COLUMN ny1 INTEGER;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='nx2') THEN
                    ALTER TABLE character_colors ADD COLUMN nx2 INTEGER;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='character_colors' AND column_name='ny2') THEN
                    ALTER TABLE character_colors ADD COLUMN ny2 INTEGER;
                END IF;
            END $$;
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detected_faces (
                id SERIAL PRIMARY KEY,
                user_email TEXT NOT NULL,
                project_id TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                color_name TEXT,
                color_hex TEXT,
                color_bgr TEXT,
                embedding FLOAT[],
                nx1 INTEGER,
                ny1 INTEGER,
                nx2 INTEGER,
                ny2 INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Add color columns if they don't exist
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='detected_faces' AND column_name='color_name') THEN
                    ALTER TABLE detected_faces ADD COLUMN color_name TEXT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='detected_faces' AND column_name='color_hex') THEN
                    ALTER TABLE detected_faces ADD COLUMN color_hex TEXT;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='detected_faces' AND column_name='color_bgr') THEN
                    ALTER TABLE detected_faces ADD COLUMN color_bgr TEXT;
                END IF;
            END $$;
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()

def record_character_color(user_email, project_id, character_name, embedding=None, storyboard_number=None, grid_number=None, nx1=None, ny1=None, nx2=None, ny2=None, table_name="character_colors"):
    """
    Records the character color mapping. Assigns the next available color from the palette.
    If the character already has a color for this project, it returns that mapping.
    """
    if not user_email or not project_id:
        return None

    # Only init the default table
    if table_name == "character_colors":
        init_db()
        
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if character already has a color assigned in this project (from any grid)
        cur.execute(f"""
            SELECT color_name, color_hex, color_bgr FROM {table_name} 
            WHERE user_email = %s AND project_id = %s AND character_name = %s
            LIMIT 1
        """, (user_email, project_id, character_name))
        existing_color = cur.fetchone()
        
        if existing_color:
            color_name = existing_color['color_name']
            color_hex = existing_color['color_hex']
            color_bgr = existing_color['color_bgr']
        else:
            # Get count of DISTINCT characters for this project to determine next color
            cur.execute(f"""
                SELECT COUNT(DISTINCT character_name) FROM {table_name} 
                WHERE user_email = %s AND project_id = %s
            """, (user_email, project_id))
            count = cur.fetchone()['count']
            
            color_idx = count % len(COLOR_PALETTE)
            color = COLOR_PALETTE[color_idx]
            
            color_name = color['name']
            color_hex = color['hex']
            color_bgr = f"({color['bgr'][0]},{color['bgr'][1]},{color['bgr'][2]})"

        # Upsert the record for this specific grid
        cur.execute(f"""
            INSERT INTO {table_name} (
                user_email, project_id, character_name, color_name, color_hex, color_bgr, 
                embedding, storyboard_number, grid_number, nx1, ny1, nx2, ny2
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT character_colors_unique_grid
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                nx1 = EXCLUDED.nx1,
                ny1 = EXCLUDED.ny1,
                nx2 = EXCLUDED.nx2,
                ny2 = EXCLUDED.ny2
            RETURNING color_name, color_hex, color_bgr
        """, (
            user_email, project_id, character_name, 
            color_name, color_hex, color_bgr, 
            embedding.tolist() if embedding is not None else None,
            storyboard_number, grid_number,
            nx1, ny1, nx2, ny2
        ))
        
        result = cur.fetchone()
        conn.commit()
        return result
    finally:
        cur.close()
        conn.close()

def get_project_characters(user_email, project_id, storyboard_number=None, grid_number=None, table_name="character_colors"):
    """Retrieves all registered characters for a specific project, optionally filtered by storyboard/grid."""
    if not user_email or not project_id:
        return []

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query = f"""
            SELECT character_name, color_name, color_hex, color_bgr, embedding, nx1, ny1, nx2, ny2
            FROM {table_name} 
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
def clear_grid_characters(user_email, project_id, storyboard_number, grid_number, table_name="character_colors"):
    """Deletes all character records for a specific grid to allow for a clean re-enhancement."""
    print(f"[DB] Clearing records in {table_name} for {user_email}/{project_id} | S{storyboard_number}-G{grid_number}")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"""
            DELETE FROM {table_name}
            WHERE user_email = %s 
              AND project_id = %s 
              AND storyboard_number = %s 
              AND grid_number = %s
        """, (user_email, project_id, storyboard_number, grid_number))
        count = cur.rowcount
        conn.commit()
        print(f"[DB] Deleted {count} stale character records.")
        return True
    finally:
        cur.close()
        conn.close()

def clear_detected_faces(user_email, project_id):
    """Deletes all detected faces for a specific user and project."""
    init_db()
    print(f"[DB] Clearing detected faces for {user_email}/{project_id}")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            DELETE FROM detected_faces
            WHERE user_email = %s AND project_id = %s
        """, (user_email, project_id))
        count = cur.rowcount
        conn.commit()
        print(f"[DB] Deleted {count} stale detected face records.")
        return True
    finally:
        cur.close()
        conn.close()

def record_detected_face(user_email, project_id, face_index, color_name, color_hex, color_bgr, embedding, nx1, ny1, nx2, ny2):
    """Records a single detected face."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO detected_faces (
                user_email, project_id, face_index, color_name, color_hex, color_bgr, embedding, nx1, ny1, nx2, ny2
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            user_email, project_id, face_index, color_name, color_hex, color_bgr,
            embedding.tolist() if embedding is not None else None,
            nx1, ny1, nx2, ny2
        ))
        result = cur.fetchone()
        conn.commit()
        return result['id'] if result else None
    finally:
        cur.close()
        conn.close()

def get_detected_faces(user_email, project_id):
    """Retrieves all detected faces for a specific project."""
    if not user_email or not project_id:
        return []

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT face_index, color_name, color_hex, color_bgr, embedding, nx1, ny1, nx2, ny2
            FROM detected_faces
            WHERE user_email = %s AND project_id = %s
            ORDER BY face_index ASC
        """, (user_email, project_id))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()
