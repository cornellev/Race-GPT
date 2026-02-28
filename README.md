**Current Data Flow**
Model: phi-4 mini
-> Race Engineer Dashboard sends current data (~5sec) via Websockets
-> Creates baseline telemtry JSON file
-> Extracts summary statistics from current telemtry data
-> Pre-check identifies any major outliers

**Run & Test the Model:**

ollama create cev-efficiency-engineer -f Modelfile

python write_telemetry.py <Optional: filepath>

python main.py

python test_ws.py
