import os
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
TEXT_MODEL = os.getenv("TEXT_MODEL_DEPLOYMENT", "")
VISION_MODEL = os.getenv("VISION_MODEL_DEPLOYMENT") or TEXT_MODEL


def client():
    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
        )
    endpoint = AOAI_ENDPOINT.removesuffix("/openai/v1").rstrip("/")
    return OpenAI(
        api_key=AOAI_KEY,
        base_url=f"{endpoint}/openai/v1/"
    )


def text_response(prompt, system="You are a helpful travel document assistant."):
    if not TEXT_MODEL:
        raise RuntimeError("Set TEXT_MODEL_DEPLOYMENT in .env")
    response = client().responses.create(
        model=TEXT_MODEL,
        instructions=system,
        input=prompt
    )
    return response.output_text


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/text-analysis")
def text_analysis():
    return render_template("text_analysis.html")


@app.post("/api/text-analysis")
def api_text_analysis():
    try:
        text = (request.json or {}).get("text", "").strip()
        if not text:
            return jsonify(error="Enter travel-related text."), 400

        prompt = f"""Analyze the following travel-related text.

Return these sections:
1. Document/Travel Type
2. Important Details
3. Dates
4. Locations
5. People/Organizations
6. Requirements or Actions
7. Warnings / Missing Information
8. Short Summary

Do not invent information. If something is not present, say "Not mentioned".

TEXT:
{text}"""

        return jsonify(result=text_response(prompt))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/image-analysis")
def image_analysis():
    return render_template("image_analysis.html")


@app.post("/api/image-analysis")
def api_image_analysis():
    try:
        f = request.files.get("image")
        if not f:
            return jsonify(error="Upload a travel document image."), 400

        data = base64.b64encode(f.read()).decode()
        mime = f.mimetype or "image/jpeg"

        prompt = """Analyze this travel document image carefully.

Try to identify the document type (for example passport, visa, ticket,
boarding pass, hotel booking, travel permit, ID, or other travel document).

Extract only information that is clearly visible:
- Name
- Document number (mask or partially hide sensitive numbers in your answer)
- Date of birth if visible
- Issue date
- Expiry date
- Travel dates
- Destination/origin
- Airline or transport details
- Visa type/status if visible
- Other important visible details

Then provide:
1. Document Type
2. Extracted Details
3. Important Dates
4. Possible Missing/Unclear Information
5. Simple Summary

Do not guess unclear text. Do not claim that a document is genuine,
valid, or legally sufficient. This is an informational analysis only."""

        response = client().responses.create(
            model=VISION_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image",
                     "image_url": f"data:{mime};base64,{data}"}
                ]
            }]
        )

        return jsonify(result=response.output_text)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/vision")
def vision():
    return render_template("vision.html")


@app.post("/api/vision")
def api_vision():
    try:
        f = request.files.get("image")
        prompt = request.form.get(
            "prompt",
            "Describe this travel-related image and identify useful visual details."
        ).strip()

        if not f:
            return jsonify(error="Upload an image."), 400

        data = base64.b64encode(f.read()).decode()
        mime = f.mimetype or "image/jpeg"

        response = client().responses.create(
            model=VISION_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image",
                     "image_url": f"data:{mime};base64,{data}"}
                ]
            }]
        )

        return jsonify(result=response.output_text)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/health")
def health():
    return jsonify(
        text_model=bool(TEXT_MODEL),
        vision_model=bool(VISION_MODEL)
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
