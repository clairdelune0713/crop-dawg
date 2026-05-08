import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from db import record_character_color, get_project_characters

app = FastAPI(title="Face Head Cropper API")

# Initialize InsightFace globally for performance
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(img):
    """Detects the largest face in a CV2 image and returns its embedding."""
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    # Sort by bbox area to find the main character (largest face)
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].embedding

def get_crop_coords(img, face, padding=0.6):
    """Calculates the coordinates for the head crop."""
    h, w, _ = img.shape
    bbox = face.bbox.astype(int)
    
    x1, y1, x2, y2 = bbox
    fw = x2 - x1
    fh = y2 - y1
    
    # Add padding
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Head crop usually needs more top padding
    side = max(fw, fh) * (1 + padding)
    
    nx1 = int(max(0, cx - side // 2 * 0.8)) # Narrow horizontal margin
    ny1 = int(max(0, cy - side // 2 * 1.3)) # Shift up for head
    nx2 = int(min(w, cx + side // 2 * 0.8)) # Narrow horizontal margin
    ny2 = int(min(h, cy + side // 2 * 0.9)) # Include more of the bottom
    
    return nx1, ny1, nx2, ny2

def crop_head(img, face, padding=0.6):
    """Crops the head from the image based on the face bounding box."""
    nx1, ny1, nx2, ny2 = get_crop_coords(img, face, padding)
    crop = img[ny1:ny2, nx1:nx2]
    return crop

@app.post("/crop")
async def crop_character(
    original: UploadFile = File(...), 
    character: UploadFile = File(...),
    user_email: str = Form(...),
    project_id: str = Form(...),
    storyboard_number: int = Form(...),
    grid_number: int = Form(...)
):
    # Read files
    try:
        original_bytes = await original.read()
        character_bytes = await character.read()
        
        nparr_orig = np.frombuffer(original_bytes, np.uint8)
        nparr_char = np.frombuffer(character_bytes, np.uint8)
        
        original_img = cv2.imdecode(nparr_orig, cv2.IMREAD_COLOR)
        character_img = cv2.imdecode(nparr_char, cv2.IMREAD_COLOR)
        
        if original_img is None or character_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")

    # Get reference embedding
    emb_ref = get_face_embedding(character_img)
    if emb_ref is None:
        raise HTTPException(status_code=400, detail="No face detected in the character portrait")

    # Detect all faces in original photo
    faces = face_app.get(original_img)
    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No faces detected in the original photo")

    threshold = 0.3
    best_match = None
    best_sim = -1

    for face in faces:
        # Cosine similarity
        sim = np.dot(emb_ref, face.embedding) / (np.linalg.norm(emb_ref) * np.linalg.norm(face.embedding))
        if sim > threshold and sim > best_sim:
            best_sim = sim
            best_match = face

    if best_match is None:
        raise HTTPException(status_code=404, detail="Character not found in the original photo")

    # Crop
    cropped_img = crop_head(original_img, best_match)
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', cropped_img)
    io_buf = BytesIO(buffer)
    
    # Record in DB (All info is now required)
    char_name = os.path.splitext(character.filename)[0] if character.filename else "unknown"
    record_character_color(
        user_email, project_id, char_name, 
        embedding=emb_ref, 
        storyboard_number=storyboard_number, 
        grid_number=grid_number
    )

    return StreamingResponse(io_buf, media_type="image/png")

@app.post("/fill-image")
async def get_fill_image(
    original: UploadFile = File(...),
    user_email: str = Form(...),
    project_id: str = Form(...),
    storyboard_number: int = Form(...),
    grid_number: int = Form(...)
):
    # Read original image
    try:
        original_bytes = await original.read()
        nparr_orig = np.frombuffer(original_bytes, np.uint8)
        original_img = cv2.imdecode(nparr_orig, cv2.IMREAD_COLOR)
        if original_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Get all characters for this project, optionally filtered by storyboard/grid
    characters = get_project_characters(
        user_email, project_id, 
        storyboard_number=storyboard_number, 
        grid_number=grid_number
    )
    if not characters:
        # Return original image if no characters found
        _, buffer = cv2.imencode('.png', original_img)
        return StreamingResponse(BytesIO(buffer), media_type="image/png")

    # Detect faces in original photo
    original_faces = face_app.get(original_img)
    if len(original_faces) == 0:
        # Return original image if no faces detected
        _, buffer = cv2.imencode('.png', original_img)
        return StreamingResponse(BytesIO(buffer), media_type="image/png")

    fill_img = original_img.copy()
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    threshold = 0.3

    for char in characters:
        emb_stored = np.array(char['embedding'])
        best_match = None
        best_sim = -1

        for face in original_faces:
            sim = np.dot(emb_stored, face.embedding) / (np.linalg.norm(emb_stored) * np.linalg.norm(face.embedding))
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_match = face

        if best_match:
            nx1, ny1, nx2, ny2 = get_crop_coords(original_img, best_match)
            
            # Parse color_bgr string "(b,g,r)"
            color_str = char['color_bgr'].strip('()')
            color = tuple(map(int, color_str.split(',')))
            
            # Create a mask for the current rectangle
            current_rect_mask = np.zeros_like(mask)
            cv2.rectangle(current_rect_mask, (nx1, ny1), (nx2, ny2), 255, -1)
            
            # Only draw where the global mask is empty
            draw_mask = cv2.bitwise_and(current_rect_mask, cv2.bitwise_not(mask))
            fill_img[draw_mask > 0] = color
            
            # Update global mask
            mask = cv2.bitwise_or(mask, current_rect_mask)

    # Encode to PNG
    _, buffer = cv2.imencode('.png', fill_img)
    return StreamingResponse(BytesIO(buffer), media_type="image/png")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
