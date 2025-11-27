from io import BytesIO
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from zipfile import ZipFile, ZIP_DEFLATED

from bss import separate_images
from PIL import Image

app = FastAPI(title="BSS Image ICA API")


@app.post("/separate")
async def separate_endpoint(files: List[UploadFile] = File(...)):
    """Upload multiple images, process them together using ICA, and return results.

    - Expect at least 2 uploaded files.
    - Calls `separate_images` once with the list of uploaded files.
    - Returns a ZIP file that contains:
        mixed_1.png, mixed_2.png, ..., recovered_1.png, recovered_2.png, ...
    """
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images are required.")

    # Read all uploaded files into memory (bytes) and prepare file-like objects
    original_bytes = []
    file_objs = []
    for f in files:
        f.file.seek(0)
        data = f.file.read()
        original_bytes.append(data)
        # create a fresh BytesIO for the separation routine (so reads don't clash)
        file_objs.append(BytesIO(data))

    # Call the separation pipeline once for all uploaded files
    try:
        result = separate_images(file_objs)
    except Exception as e:
        # Surface processing errors as HTTP 500 with the original message
        raise HTTPException(status_code=500, detail=f"Separation failed: {e}")

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="separate_images returned an unexpected result type")

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        # Save recovered images produced by separate_images
        # Keys expected like 'recovered_1', 'recovered_2', ...
        for key in sorted(result.keys()):
            img = result[key]
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"{key}.png", img_bytes.read())

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition": "attachment; filename=separated_images.zip"
        },
    )


@app.get("/")
async def root():
    return {"message": "BSS Image ICA API is running. POST images to /separate"}
