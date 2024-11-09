from fastapi import FastAPI, HTTPException, File, UploadFile
import pickle
import numpy as np
from PIL import Image
import io

# Define the image transformation function directly
def image_to_vector(image):
    try:
        # Resize the image to 32x32 and convert to grayscale
        image = image.resize((32, 32)).convert("L")
        
        # Convert the image to a numpy array and flatten it into a vector
        image_array = np.array(image)
        vector = image_array.flatten()
        
        return vector
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None

# Load the model and scaler
try:
    with open("/app/artifacts/model.pickle", "rb") as f:
        model = pickle.load(f)
    
    with open("/app/artifacts/scaler.pickle", "rb") as f1:
        scaler = pickle.load(f1)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None

# Create the FastAPI application
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if the file is an image (this can be expanded for other formats)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read the image file
        image_data = await file.read()
        
        # Convert the binary data to a PIL Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Transform the image to a vector
        features = image_to_vector(image)
        
        if features is None:
            raise HTTPException(status_code=500, detail="Error in image processing.")
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make a prediction with the model
        prediction = model.predict(features_scaled)
        
        # Check if the prediction is a string (e.g., 'cat', 'dog', etc.)
        if isinstance(prediction[0], str):
            return {"prediction": prediction[0]}
        
        # If the prediction is numeric, return it as an integer
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
