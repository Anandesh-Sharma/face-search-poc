from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Initialize FastAPI
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a specific domain or list of domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Rekognition client
rekognition_client = boto3.client('rekognition', region_name='us-east-1')

# AWS Rekognition Collection ID
COLLECTION_ID = "satschel_faces_v1"

# Ensure Rekognition collection is created
def create_collection(collection_id):
    try:
        rekognition_client.create_collection(CollectionId=collection_id)
        print(f"Collection '{collection_id}' created.")
    except rekognition_client.exceptions.ResourceAlreadyExistsException:
        print(f"Collection '{collection_id}' already exists.")
    except (BotoCoreError, ClientError) as e:
        print(f"Error creating collection: {e}")

create_collection(COLLECTION_ID)


@app.post("/enroll")
async def enroll_face(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        # Read image file as bytes
        image_bytes = await file.read()

        # Index the face in the Rekognition collection
        response = rekognition_client.index_faces(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            ExternalImageId=user_id,  # Use provided user_id to associate with the face
            DetectionAttributes=['ALL']
        )

        if len(response['FaceRecords']) == 0:
            raise HTTPException(status_code=404, detail="No face detected in the image.")
        
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }

        return JSONResponse(content={"message": "Face enrolled successfully", "FaceRecords": response['FaceRecords']}, headers=headers)

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    try:
        # Read image file as bytes
        image_bytes = await file.read()

        # Search for the face in the Rekognition collection
        response = rekognition_client.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            MaxFaces=1,
            FaceMatchThreshold=85  # Confidence threshold
        )

        if len(response['FaceMatches']) == 0:
            return {"message": "No matching face found"}
        
        # Disable client-side caching
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }

        return JSONResponse(content={"message": "Match found", "FaceMatches": response['FaceMatches']}, headers=headers)

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=str(e))




