**Current Data Flow**

Model: phi-4 mini
-> Race Engineer Dashboard sends current data (~5sec) via Websockets
-> Creates baseline telemtry JSON file
-> Extracts summary statistics from current telemtry data
    -> Statistics include mean, std, p05, p95, covariance matrix between all columns
-> Pre-check identifies any critical problems (major dips/peaks)

**System Prompt:**

You are an AI efficiency engineer for the Cornell Electric Vehicles car efficiency racing team that competes at the Shell Eco-Marathon.

Your task is to verbalize the PRECHECK_DECISION into exactly one clear sentence.
Do not invent new issues or add extra details beyond the decision provided.
If PRECHECK_DECISION already contains the full message, lightly rephrase for clarity without changing meaning.
Your output must be exactly one sentence and avoid filler words.

**Structure of User Prompt:**

-> PRECHECK_DECISION 
-> BASELINE_TELEMETRY (json of summary statistics)
-> CURRENT_SUMMARY (json of summary statistics)

The Race-GPT model will output a one-line response based on the pre-check decision and current summary data compared to baseline telemetry data.

**Run & Test the Model:**

ollama create cev-efficiency-engineer -f Modelfile

python write_telemetry.py <Optional: filepath>

python main.py

python test_ws.py
