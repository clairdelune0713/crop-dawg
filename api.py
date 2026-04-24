import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

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

def crop_head(img, face, padding=0.6):
    """Crops the head from the image based on the face bounding box."""
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
    
    nx1 = int(max(0, cx - side // 2))
    ny1 = int(max(0, cy - side // 2 * 1.3)) # Shift up for head
    nx2 = int(min(w, cx + side // 2))
    ny2 = int(min(h, cy + side // 2 * 0.7)) # Shift up
    
    crop = img[ny1:ny2, nx1:nx2]
    return crop

@app.post("/crop")
async def crop_character(original: UploadFile = File(...), character: UploadFile = File(...)):
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

    threshold = 0.4
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
    
    return StreamingResponse(io_buf, media_type="image/png")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
