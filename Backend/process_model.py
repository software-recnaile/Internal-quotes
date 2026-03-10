from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import asyncio
import trimesh

router = APIRouter()

async def process_single_stl(file: UploadFile):
    """Process an individual STL file using trimesh and return its volume."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        mesh = trimesh.load(tmp_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return {
                "filename": file.filename,
                "error": "File does not contain a single valid mesh.",
                "reason": "non watertighted material"
            }

        volume_mm3 = mesh.volume

        # If volume is less than or equal to 0 mm³, it's an error file
        if volume_mm3 <= 0:
            return {
                "filename": file.filename,
                "volume_cm3": 0,
                "reason": "error file"
            }

        # If volume is between 1 mm³ and 1000 mm³, set to 1 cm³
        if 1 <= volume_mm3 <= 1000:
            volume_cm3 = 1
        else:
            volume_cm3 = round(volume_mm3 / 1000)

        if mesh.is_watertight:
            # Watertight mesh: only filename and volume
            return {
                "filename": file.filename,
                "volume_cm3": volume_cm3
            }
        else:
            # Non-watertight mesh: include reason
            return {
                "filename": file.filename,
                "volume_cm3": volume_cm3,
                "reason": "non watertighted material"
            }

    except Exception as e:
        return {
            "filename": file.filename,
            "error": str(e),
            "reason": "non watertighted material"
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass



@router.post("/upload-multiple-stl/")
async def upload_multiple_stl(files: list[UploadFile] = File(...)):
    """
    Process multiple STL files concurrently using trimesh.
    Returns individual results for each file.
    """
    tasks = [process_single_stl(file) for file in files]
    results = await asyncio.gather(*tasks)

    response = {
        "processed": [],
        "errors": []
    }

    for result in results:
        if "error" in result:
            response["errors"].append(result)
        else:
            response["processed"].append(result)

    return JSONResponse(content=response)
