from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Initialize FastAPI
app = FastAPI()

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

        return {"message": "Face enrolled successfully", "FaceRecords": response['FaceRecords']}

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

        return {"message": "Match found", "FaceMatches": response['FaceMatches']}

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=str(e))




