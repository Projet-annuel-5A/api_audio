# Audio Emotions Analysis module

## Overview
The Audio Emotions Analysis module of the **Interviewz** application is responsible for analyzing audio segments from interviews and determining the emotional content within these segments. The module uses FastAPI for serving the endpoints, PyTorch and torchaudio for audio processing, and a Supabase database for storing and retrieving data.

## Directory Structure
The module consists of several Python files organized as follows:
```plaintext
audio/
├── app.py
├── audioEmotions.py
├── utils/
│   ├── models.py
│   ├── utils.py
```

## Components

### FastAPI Application (app.py)
Initializes a FastAPI application.

#### API Endpoints

```fastAPI
@app.get("/health")
"""
Returns the health status of the API. 
Description: Endpoint for checking the health status of the application.
Response: Returns a JSON object with the status "ok".
"""
```
```fastAPI
@app.post("/analyse_audio")
"""
Processes an audio file to analyze emotions.
Parameters:
    session_id (int): The session ID related to the audio file.
    interview_id (int): The interview ID of the audio file.
Returns:
    dict: Status message indicating the outcome of the operation.
Raises:
    HTTPException: An exception with status code 500 if processing fails.
"""
```

### AudioEmotions (audioEmotions.py):
Handles the extraction of audio segments from storage and predicts emotions using pre-trained models.
Utilizes torchaudio for audio handling and transformers for emotion classification.

### Utilities (utils/utils.py): 
Provides methods for logging, configuration management, file operations, and database interactions.
Manages connections to both Supabase for data handling and S3 buckets for file storage.

### Models (utils/models.py):

Manages the loading and usage of machine learning models for audio classification.
Ensures models are loaded once using singleton pattern to optimize resources.