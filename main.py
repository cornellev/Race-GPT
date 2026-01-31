from fastapi import FastAPI
from pydantic import BaseModel
from ollama import chat

app = FastAPI()


class TelemetryIn(BaseModel):
    csv: str


@app.post("/analyze")
def analyze(data: TelemetryIn):
    prompt = f"""
    CURRENT_TELEMETRY:
    {data.csv}
    """.strip()

    response = chat(
        model="cev-efficiency-engineer",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        keep_alive=600,
        options={
            "temperature": 0.05,
            "top_p": 0.7,
            "repeat_penalty": 1.15,
            "num_predict": 32,
            "stop": ["\n"],
        },
    )

    text = response["message"]["content"].strip()

    """
    # Absolute safety clamp (should almost never trigger)
    if "." in text:
        text = text.split(".")[0].strip() + "."
    else:
        text = text.strip() + "."

        
    """
    return {"verdict": text}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)