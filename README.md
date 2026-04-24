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

**Response**:
- **Success (200)**: Returns the cropped image file directly as a PNG.
- **Error (400/404)**: Returns a JSON object with the error detail.

**Example using `curl`**:
```bash
curl -X POST "http://localhost:8000/crop" \
  -F "original=@original/ss.png" \
  -F "character=@character/Cousin_Sean-0331.png" \
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

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('output.png', 'wb') as f:
        f.write(response.content)
else:
    print(f"Error: {response.json()['detail']}")
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
