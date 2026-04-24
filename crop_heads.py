import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
import glob

def get_face_embedding(app, img_path):
    """Detects the largest face in an image and returns its embedding."""
    img_pil = Image.open(img_path)
    img_pil = ImageOps.exif_transpose(img_pil)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return None
    
    faces = app.get(img)
    if len(faces) == 0:
        print(f"Warning: No faces detected in {img_path}")
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

def main():
    # Initialize InsightFace
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1280, 1280))

    # Define directories
    original_dir = 'original'
    character_dir = 'character'
    output_dir = 'crops'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find original photo
    original_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        original_files.extend(glob.glob(os.path.join(original_dir, ext)))
    
    if not original_files:
        print(f"Error: No image found in {original_dir}/")
        return
    
    original_path = original_files[0]
    print(f"Using original photo: {original_path}")
    img_pil = Image.open(original_path)
    img_pil = ImageOps.exif_transpose(img_pil)
    original_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if original_img is None:
        print("Error: Could not read original image.")
        return

    # Detect all faces in original photo once
    print("Analyzing original photo for faces...")
    original_faces = app.get(original_img)
    print(f"Found {len(original_faces)} faces in original photo.")

    # Find character portraits
    character_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        character_files.extend(glob.glob(os.path.join(character_dir, ext)))

    if not character_files:
        print(f"Error: No character portraits found in {character_dir}/")
        return

    threshold = 0.4 # Cosine similarity threshold

    for char_path in character_files:
        char_name = os.path.splitext(os.path.basename(char_path))[0]
        print(f"\nProcessing character: {char_name}")
        
        emb = get_face_embedding(app, char_path)
        if emb is None:
            print(f"Skipping {char_name}: No face detected in portrait.")
            continue

        best_match = None
        best_sim = -1

        for face in original_faces:
            # Cosine similarity
            sim = np.dot(emb, face.embedding) / (np.linalg.norm(emb) * np.linalg.norm(face.embedding))
            
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_match = face

        if best_match:
            print(f"Match found for {char_name} (sim: {best_sim:.4f}). Cropping...")
            crop = crop_head(original_img, best_match)
            output_name = os.path.join(output_dir, f"{char_name}-crop.png")
            cv2.imwrite(output_name, crop)
            print(f"Saved {output_name}")
        else:
            print(f"No match found for {char_name} in the original photo.")

if __name__ == "__main__":
    main()
