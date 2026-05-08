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
# High-res model for complex scenes
face_app_1280 = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app_1280.prepare(ctx_id=0, det_size=(1280, 1280))

# Standard model for portraits and fallback
face_app_640 = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app_640.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(img):
    """Detects the largest face in a CV2 image and returns its embedding."""
    h, w, _ = img.shape
    print(f"[get_face_embedding] Input image size: {w}x{h}")
    
    # Try with 1280x1280
    faces = face_app_1280.get(img, det_thresh=0.4)
    
    # Fallback to 640x640
    if len(faces) == 0:
        print("[get_face_embedding] No face found at 1280x1280, trying 640x640...")
        faces = face_app_640.get(img, det_thresh=0.4)
        
    if len(faces) == 0:
        print("[get_face_embedding] Still no face found after fallback.")
        return None
        
    print(f"[get_face_embedding] Found {len(faces)} faces. Selecting largest.")
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

    # Detect all faces in original photo (Try 1280 first)
    faces = face_app_1280.get(original_img, det_thresh=0.4)
    
    def find_best_match(target_faces, ref_emb, label=""):
        best_m = None
        best_s = -1
        second_s = -1
        if label:
            print(f"  [Matching {label}] against {len(target_faces)} faces:")
        for i, face in enumerate(target_faces):
            sim = np.dot(ref_emb, face.embedding) / (np.linalg.norm(ref_emb) * np.linalg.norm(face.embedding))
            if label:
                print(f"    Face {i}: similarity = {sim:.4f}")
            if sim > best_s:
                second_s = best_s
                best_s = sim
                best_m = face
            elif sim > second_s:
                second_s = sim
        return best_m, best_s, second_s

    best_match, best_sim, second_best_sim = find_best_match(faces, emb_ref, label="1280x1280")
    
    # Check if we have a valid match
    threshold = 0.25
    is_match = (best_match is not None) and (
        (best_sim >= threshold) or \
        (best_sim >= 0.20 and best_sim > (second_best_sim + 0.05)) or \
        (len(faces) <= 2 and best_sim >= 0.18)
    )

    # If no match, try detecting faces in original at 640x640 (Fallback detection)
    if not is_match:
        print(f"[crop] No threshold met at 1280x1280 (Best: {best_sim:.4f}). Retrying original detection at 640x640...")
        faces_640 = face_app_640.get(original_img, det_thresh=0.4)
        if len(faces_640) > 0:
            m_640, s_640, ss_640 = find_best_match(faces_640, emb_ref, label="640x640 Fallback")
            
            # If 640 found a better candidate than 1280, update our globals even if it doesn't meet threshold
            if s_640 > best_sim:
                best_match, best_sim, second_best_sim = m_640, s_640, ss_640

            is_m_640 = (s_640 >= threshold) or \
                       (s_640 >= 0.20 and s_640 > (ss_640 + 0.05)) or \
                       (len(faces_640) <= 2 and s_640 >= 0.18)
            
            if is_m_640:
                print(f"[crop] Found match in 640x640 fallback! Sim: {s_640:.4f}")
                is_match = True

    if not is_match:
        if best_match:
            print(f"[crop] FORCE MATCH: No threshold met. Picking highest sim across all passes: {best_sim:.4f}")
            is_match = True
        else:
            print(f"No faces detected for {character.filename}.")
            raise HTTPException(status_code=404, detail=f"No faces detected in the original photo")

    print(f"Found match for {character.filename} with sim: {best_sim:.4f}")

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

    # Detect faces in original photo (Try 1280 first)
    faces_1280 = face_app_1280.get(original_img, det_thresh=0.4)
    faces_640 = face_app_640.get(original_img, det_thresh=0.4)
    
    fill_img = original_img.copy()
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    threshold = 0.25

    def find_best_match(target_faces, ref_emb):
        best_m = None
        best_s = -1
        second_s = -1
        for i, face in enumerate(target_faces):
            sim = np.dot(ref_emb, face.embedding) / (np.linalg.norm(ref_emb) * np.linalg.norm(face.embedding))
            if sim > best_s:
                second_s = best_s
                best_s = sim
                best_m = face
            elif sim > second_s:
                second_s = sim
        return best_m, best_s, second_s

    print(f"Generating fill image for {len(characters)} characters...")
    for char in characters:
        char_name = char['character_name']
        emb_stored = np.array(char['embedding'])
        
        # Try matching with 1280 faces first
        best_match, best_sim, second_best_sim = find_best_match(faces_1280, emb_stored, label=f"1280 {char_name}")
        is_match = (best_match is not None) and (
            (best_sim >= threshold) or \
            (best_sim >= 0.20 and best_sim > (second_best_sim + 0.05)) or \
            (len(faces_1280) <= 2 and best_sim >= 0.18)
        )

        # Fallback to 640 faces if not matched
        if not is_match and len(faces_640) > 0:
            print(f"    [fill] No match for {char_name} at 1280x1280. Retrying at 640x640...")
            m_640, s_640, ss_640 = find_best_match(faces_640, emb_stored, label=f"640 {char_name} Fallback")
            if (s_640 >= threshold) or \
               (s_640 >= 0.20 and s_640 > (ss_640 + 0.05)) or \
               (len(faces_640) <= 2 and s_640 >= 0.18):
                best_match, best_sim, second_best_sim = m_640, s_640, ss_640
                is_match = True

        if not is_match and best_match:
            print(f"    [fill] FORCE MATCH for {char_name} with sim: {best_sim:.4f}")
            is_match = True

        if is_match:
            print(f"    [fill] Final match for {char_name} with sim: {best_sim:.4f}")
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
