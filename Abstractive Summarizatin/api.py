from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from abstractive import *
from ocr import *
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Replace "*" with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
# class TextInput(BaseModel):
#     text: str

# Summarization API
@app.post("/summarize/")
async def summarize_text(input_text: str = Form(None), file: UploadFile = File(None), file_type: str = Form(None)):
    temp_dir = './temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    if file:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        if file_type == "pdf":
            input_text = extract_text_from_pdf(file_path)

        os.remove(file_path)
    
    if not input_text:
        return JSONResponse({"error": "No input text or file provided."})
    

    # text = input_data.text
    summaries, evaluation_metrics = evaluate_summarization(models, input_text)
    
    return {
        "summaries": summaries,
        "evaluation_metrics": evaluation_metrics
    }

# To run the app:
# uvicorn main:app --reload
