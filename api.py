import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from db import record_character_color, get_project_characters, clear_grid_characters

app = FastAPI(title="Face Head Cropper API")

# High-res model for complex scenes
face_app_1280 = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app_1280.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.1) # Very low thresh to catch everything

# Standard model for portraits and fallback
face_app_640 = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app_640.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

def get_face_embedding(img):
    """Detects the largest face in a CV2 image and returns its embedding."""
    h, w, _ = img.shape
    print(f"[get_face_embedding] Input image size: {w}x{h}")
    
    # Try with 1280x1280
    faces = face_app_1280.get(img)
    
    # Fallback to 640x640
    if len(faces) == 0:
        print("[get_face_embedding] No face found at 1280x1280, trying 640x640...")
        faces = face_app_640.get(img)
        
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
    
    nx1 = int(max(0, cx - side // 2 * 1.0)) # Wider horizontal margin
    ny1 = int(max(0, cy - side // 2 * 1.3)) # Shift up for head
    nx2 = int(min(w, cx + side // 2 * 1.0)) # Wider horizontal margin
    ny2 = int(min(h, cy + side // 2 * 1.6)) # Include shoulders (more bottom)
    
    return nx1, ny1, nx2, ny2

def find_best_match(target_faces, ref_emb, label=""):
    """Finds the best matching face among candidates."""
    best_m = None
    best_s = -1
    second_s = -1
    if label:
        print(f"  [Matching {label}] against {len(target_faces)} faces:")
    for i, face in enumerate(target_faces):
        # Cosine similarity
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

    # 1. Exhaustive Multi-Scale Detection
    all_faces = []
    scales = [(1280, 1280), (960, 960), (640, 640)]
    
    for sw, sh in scales:
        print(f"[crop] Detecting faces at {sw}x{sh}...")
        # We reuse the apps but change det_size dynamically if needed, 
        # but for simplicity we'll just use the two prepared ones for now 
        # and maybe add a third if it helps.
        app = face_app_1280 if sw > 640 else face_app_640
        # InsightFace apps det_size is set during prepare, so we just use the two we have.
        curr_faces = app.get(original_img)
        print(f"    Found {len(curr_faces)} candidates.")
        all_faces.extend(curr_faces)
        if len(curr_faces) > 0 and sw == 1280:
            # If we found faces at high res, we might still want to try lower res 
            # for smaller faces that might have been missed by high-res stride.
            pass

    # Deduplicate faces based on bounding box overlap (IOU)
    def get_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

    unique_faces = []
    for f in all_faces:
        is_duplicate = False
        for uf in unique_faces:
            if get_iou(f.bbox, uf.bbox) > 0.5:
                # Keep the one with higher detection score
                if f.det_score > uf.det_score:
                    unique_faces.remove(uf)
                    unique_faces.append(f)
                is_duplicate = True
                break
        if not is_duplicate:
            unique_faces.append(f)

    print(f"[crop] Total unique candidates: {len(unique_faces)}")

    if not unique_faces:
        # Last resort: Try very small detection size and lowest threshold
        print("[crop] NO FACES FOUND. Last resort attempt at 320x320...")
        face_app_640.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.01)
        unique_faces = face_app_640.get(original_img)
        # Restore for next call
        face_app_640.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

    if not unique_faces:
        print(f"No faces detected for {character.filename} even after fallback.")
        raise HTTPException(status_code=404, detail=f"No faces detected in the original photo")

    # 2. Matching
    best_match, best_sim, second_best_sim = find_best_match(unique_faces, emb_ref, label="Exhaustive")
    
    # Check if we have a valid match
    threshold = 0.25
    is_match = (best_match is not None) and (
        (best_sim >= threshold) or \
        (best_sim >= 0.20 and best_sim > (second_best_sim + 0.10)) or \
        (len(unique_faces) == 1) # If only one face in the whole image, it MUST be the one
    )

    if not is_match:
        print(f"[crop] FORCE MATCH: Sim {best_sim:.4f} below threshold, but picking best candidate to satisfy 'ensure one' requirement.")
        is_match = True

    print(f"Found final match for {character.filename} with sim: {best_sim:.4f}")

    # Crop
    cropped_img = crop_head(original_img, best_match)
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', cropped_img)
    io_buf = BytesIO(buffer)
    
    # Calculate crop coordinates for DB
    nx1, ny1, nx2, ny2 = get_crop_coords(original_img, best_match)
    
    # Record in DB (All info is now required)
    char_name = os.path.splitext(character.filename)[0] if character.filename else "unknown"
    record_character_color(
        user_email, project_id, char_name, 
        embedding=emb_ref, 
        storyboard_number=storyboard_number, 
        grid_number=grid_number,
        nx1=int(nx1), ny1=int(ny1), nx2=int(nx2), ny2=int(ny2)
    )

    return StreamingResponse(io_buf, media_type="image/png")

@app.post("/fill-image")
async def get_fill_image(
    original: UploadFile = File(...),
    user_email: str = Form(...),
    project_id: str = Form(...),
    storyboard_number: int = Form(...),
    grid_number: int = Form(...),
    char_names: str = Form(None)
):
    # Read original image
    try:
        original_bytes = await original.read()
        nparr_orig = np.frombuffer(original_bytes, np.uint8)
        original_img = cv2.imdecode(nparr_orig, cv2.IMREAD_COLOR)
        if original_img is None:
            raise HTTPException(status_code=400, detail="Invalid original image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Get all characters for this project, optionally filtered by storyboard/grid
    characters = get_project_characters(
        user_email, project_id, 
        storyboard_number=storyboard_number, 
        grid_number=grid_number
    )
    
    # Filter by specific names if provided
    if char_names:
        name_list = [n.strip() for n in char_names.split(',')]
        characters = [c for c in characters if c['character_name'] in name_list]

    if not characters:
        # Return original image if no characters found
        _, buffer = cv2.imencode('.png', original_img)
        return StreamingResponse(BytesIO(buffer), media_type="image/png")

    # Detect faces in original photo (Try 1280 first)
    faces_1280 = face_app_1280.get(original_img)
    faces_640 = face_app_640.get(original_img)
    
    fill_img = original_img.copy()
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    threshold = 0.25

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
            b, g, r = map(int, color_str.split(','))
            color_bgr = (b, g, r)
            
            # Draw SOLID rectangle on fill image (thickness = -1)
            cv2.rectangle(fill_img, (nx1, ny1), (nx2, ny2), color_bgr, -1)
            # Fill mask
            cv2.rectangle(mask, (nx1, ny1), (nx2, ny2), 255, -1)
    # Encode result
    _, buffer = cv2.imencode('.png', fill_img)
    return StreamingResponse(BytesIO(buffer), media_type="image/png")

@app.post("/clear-grid")
async def clear_grid(
    user_email: str = Form(...),
    project_id: str = Form(...),
    storyboard_number: int = Form(...),
    grid_number: int = Form(...)
):
    try:
        clear_grid_characters(user_email, project_id, storyboard_number, grid_number)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
