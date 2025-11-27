from io import BytesIO
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from zipfile import ZipFile, ZIP_DEFLATED

from bss import separate_images

app = FastAPI(title="BSS Image ICA API")


@app.post("/separate")
async def separate_endpoint(files: List[UploadFile] = File(...)):
    """Upload multiple images, process them in pairs using ICA, and return results.

    - Expect an even number of files (pairs: img1, img2), (img3, img4), ...
    - For each pair we run `separate_images` from bss.py.
    - Returns a ZIP file that contains, for each pair i:
        pair{i}_mixed_1.png
        pair{i}_mixed_2.png
        pair{i}_recovered_1.png
        pair{i}_recovered_2.png
    """
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images are required.")

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        # Process in pairs: (0,1), (2,3), ...
        for i in range(0, len(files), 2):
            f1 = files[i]
            f2 = files[i + 1]

            # Reset stream positions (FastAPI UploadFile.file is a SpooledTemporaryFile)
            f1.file.seek(0)
            f2.file.seek(0)

            # Call the separation pipeline using underlying file objects
            result = separate_images(f1.file, f2.file)

            # Save all four output images to in-memory buffers and then into zip
            for key, img in result.items():
                img_bytes = BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Name: pair{index}_{key}.png (index starting from 1)
                pair_index = i // 2 + 1
                filename = f"pair{pair_index}_{key}.png"

                zip_file.writestr(filename, img_bytes.read())

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
