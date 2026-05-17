# Face Head Cropper API

This project provides an automated face cropping service that identifies a specific person from a long-shot photo using a reference portrait and returns a high-resolution crop of their head.

## Features
- **Smart Detection**: Uses InsightFace (Buffalo_L) for high-accuracy face detection and recognition.
- **Background Filtering**: Automatically ignores non-main characters.
- **Maximum Resolution**: Returns the crop in its original resolution from the source image.
- **RESTful API**: Easy integration with any frontend or service.

---

## Installation & Setup

1. **Requirements**: Python 3.12+ and `uv`.
2. **Install Dependencies**:
   ```bash
   uv sync
   ```

## Running the API
Start the server using `uv`:
```bash
uv run api
```

### Running with Docker
Alternatively, you can run the service using Docker:
```bash
docker compose up --build
```
The server will be available at `http://localhost:8000`.
The `original/`, `character/`, and `crops/` folders are mapped as volumes, so you can still use them for local processing while the container is running.

---

## API Documentation

### 1. Crop Character
**Endpoint**: `POST /crop`

Identifies a character in an original photo based on a reference portrait and returns the cropped head.

**Request (Multipart/Form-Data)**:
| Field | Type | Description |
| :--- | :--- | :--- |
| `original` | File | The original long-shot image (JPG, PNG, WebP). |
| `character` | File | The reference portrait of the person to crop. |
| `user_email` | String | (Required) Email to track character ownership. |
| `project_id` | String | (Required) Unique ID for the project. |
| `storyboard_number` | Integer | (Required) Storyboard index for the scene. |
| `grid_number` | Integer | (Required) Grid index within the storyboard. |

**Response**:
- **Success (200)**: Returns the cropped image file directly as a PNG.
- **Error (400/404)**: Returns a JSON object with the error detail.

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/crop" \
  -F "original=@original/ss.png" \
  -F "character=@character/Cousin_Sean-0331.png" \
  -F "user_email=test@example.com" \
  -F "project_id=project_001" \
  -F "storyboard_number=1" \
  -F "grid_number=4" \
  --output result.png
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/crop"
files = {
    'original': open('original/ss.png', 'rb'),
    'character': open('character/Cousin_Sean-0331.png', 'rb')
}
data = {
    'user_email': 'test@example.com',
    'project_id': 'project_001',
    'storyboard_number': 1,
    'grid_number': 4
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('output.png', 'wb') as f:
        f.write(response.content)
else:
    print(f"Error: {response.json()['detail']}")
```

### 2. Get Fill Image
**Endpoint**: `POST /fill-image`

Returns the original image with colored rectangles covering all characters that have been cropped for the given project. This endpoint uses stored facial embeddings and does not require reference portraits.

**Request (Multipart/Form-Data)**:
| Field | Type | Description |
| :--- | :--- | :--- |
| `original` | File | The original long-shot image. |
| `user_email` | String | (Required) User email. |
| `project_id` | String | (Required) Project ID. |
| `storyboard_number` | Integer | (Required) Storyboard index to visualize. |
| `grid_number` | Integer | (Required) Grid index to visualize. |

**Response**:
- **Success (200)**: Returns the visualization image directly as a PNG.

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/fill-image" \
  -F "original=@original/ss.png" \
  -F "user_email=test@example.com" \
  -F "project_id=project_001" \
  -F "storyboard_number=1" \
  -F "grid_number=4" \
  --output fill_status.png
```

### 3. Detect All Faces
**Endpoint**: `POST /detect-faces`

Detects all unique faces in a single long-shot image, crops them individually, automatically records them and their assigned aesthetic colors in the database (under the new `detected_faces` table), and returns both individual base64-encoded crops and a base64-encoded visualizer image overlaying all detected faces.

**Request (Multipart/Form-Data)**:
| Field | Type | Description |
| :--- | :--- | :--- |
| `original` | File | The original long-shot image (JPG, PNG, WebP). |
| `user_email` | String | (Required) User email. |
| `project_id` | String | (Required) Project ID. |

**Response (JSON)**:
- **Success (200)**: Returns a JSON object with `"success": true`, a `"faces"` array containing base64 crops/metadata, and `"fill_image"` containing a base64 encoded fill visualization.
  ```json
  {
    "success": true,
    "faces": [
      {
        "face_index": 0,
        "base64": "...(base64 encoded png crop)...",
        "color_name": "red",
        "color_hex": "#FF0000",
        "coords": [189, 105, 374, 395]
      },
      ...
    ],
    "fill_image": "...(base64 encoded full color-fill visualizer image)..."
  }
  ```

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/detect-faces" \
  -F "original=@original/all.png" \
  -F "user_email=test@example.com" \
  -F "project_id=project_001"
```

### 4. Get Detected Faces Fill Image
**Endpoint**: `POST /detect-faces-fill`

Returns a color-filled visualization image covering all detected faces for the given project. This endpoint streams the PNG image directly.

**Request (Multipart/Form-Data)**:
| Field | Type | Description |
| :--- | :--- | :--- |
| `original` | File | The original long-shot image. |
| `user_email` | String | (Required) User email. |
| `project_id` | String | (Required) Project ID. |

**Response**:
- **Success (200)**: Streams the visualizer PNG image directly.

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/detect-faces-fill" \
  -F "original=@original/all.png" \
  -F "user_email=test@example.com" \
  -F "project_id=project_001" \
  --output detect_faces_fill.png
```

---

## Local Script Usage (CLI)
You can still use the local batch processing script:
1. Put images in `original/` and `character/`.
2. Run:
   ```bash
   uv run crop
   ```
3. Results will be saved in the `crops/` folder.
