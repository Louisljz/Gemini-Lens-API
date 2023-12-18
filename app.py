import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

import io
import os
from dotenv import load_dotenv

import google.generativeai as genai
from trulens_eval import Feedback, Tru, OpenAI, TruBasicApp

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-pro-vision')
tru = Tru(database_url=os.getenv('DATABASE_URL'))
app = FastAPI()

def setup_feedbacks():
    openai_provider = OpenAI()
    relevance = Feedback(openai_provider.relevance).on_input_output()
    conciseness = Feedback(openai_provider.conciseness).on_output()
    return [relevance, conciseness]

def run_gemini(query, image):
    message = f'''Analyze the image and answer the question below. If the image is unrelated, 
            ignore it and answer based on textual knowledge. Keep your response as concise as possible.
            Question: {query}'''
    response = gemini_model.generate_content([message, image])
    return response.text

feedbacks = setup_feedbacks()
gemini_recorder = TruBasicApp(run_gemini, app_id="Gemini-Vision", feedbacks=feedbacks)

@app.post("/process/")
async def process(query: str, image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        pil_img = Image.open(io.BytesIO(image_data))
        with gemini_recorder as recording:
            response = gemini_recorder.app(query, pil_img)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
