## Smart Contract Malicious Behavior Detection Project

This project implements a **malicious behavior detection system for Ethereum smart contracts**.  
The end‑to‑end pipeline is:

1. User enters a contract address (with `0x`).
2. Retrieve on‑chain bytecode via an Ethereum RPC endpoint.
3. Use vandal-master (https://github.com/usyd-blockchain/vandal) to decompile bytecode into CFG facts.
4. Run a GNN model (PyTorch Geometric) on the facts to estimate maliciousness.
5. Return the result via **CLI** or **Web UI**, including *malicious score* and *model confidence*.

There are two main entrypoints:

- **CLI pipeline**: `pipeline.py`
- **Web application (Flask)**: `webapp/app.py`


## Requirements

- Python 3.8+ (virtualenv or Conda is recommended).
- OS: Windows / Linux / macOS (paths in code currently assume Windows, but can be adjusted).
- A reachable Ethereum RPC endpoint (e.g. Alchemy, Infura, or your own node).  
  The default RPC URL is configured in `pipeline.py` and can be replaced.

---

## Setup

### 1. Get the Project

Place the project folder on your machine, for example:


### 2. Create a Virtual Environment (Recommended)

On Windows:

```bash
cd c:\Users\john\Desktop\deploy
python -m venv .venv
.venv\Scripts\activate
```

On Linux / macOS:

```bash
cd /path/to/deploy
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

In the project root (where `requirements.txt` lives), run:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `web3`: Fetch contract bytecode from an Ethereum node.
- `flask`: Web application framework.
- `torch`, `torch-geometric`, `numpy`: GNN model inference and feature processing.

> Note: `torch-geometric` can require OS‑ and CUDA‑specific wheels.  
> If `pip install -r requirements.txt` fails, please follow the official PyTorch Geometric installation instructions for your environment, then re‑run the install.

### 4. Prepare Vandal and GNN Models

- Make sure the `vandal-master/` directory is complete and `bin/decompile` is executable by Python.
- Place the trained model weights in the project root, e.g.:  
  `model_fold_1.pth` ~ `model_fold_5.pth` (as used in `GNN/detect.py`).
- If you store the model files elsewhere, update the path in `GNN/detect.py`:

```python
MODEL_DIR = r"C:\Users\john\Desktop\FYP"
```

Change it to the actual directory containing your `.pth` files.

---

## Usage – CLI Pipeline (`pipeline.py`)

### Run from Command Line

From the project root, with the virtual environment activated:

```bash
python pipeline.py
```

Then follow the prompt to enter a smart contract address (with `0x`, or without – the script will prepend `0x` automatically).

Pipeline steps:

1. Fetch the target contract bytecode from the chain.  
2. Save the bytecode as a `.hex` file under `vandal-master/contracts_input_hex/`.  
3. Call Vandal’s `bin/decompile` to output CFG facts into `vandal-master/contracts_output_cfg/`.  
4. Run `GNN/detect.py` on the facts directory to detect malicious behavior.  
5. Print the detection result in the terminal.

You can also call the pipeline programmatically:

```python
from pipeline import run_pipeline

success, data = run_pipeline("0x....")
if success:
    print("Detection result:", data)
else:
    print("Error:", data)
```

---

## Usage – Web Application (`webapp/app.py`)

### Start the Web Server

From the project root, with the virtual environment activated:

```bash
cd webapp
python app.py
```

By default, the app will start on:

```text
http://127.0.0.1:5000/
```

### Web Flow

1. The front‑end (`index.html`) provides an input box for the smart contract address.  
2. The front‑end sends a request to the `/analyze` API to create a detection job; the server returns a `job_id`.  
3. The server starts a background thread that calls `pipeline.run_pipeline(address)` to execute the full pipeline.  
4. The front‑end polls `/status/<job_id>` to check job status:
   - `status = "pending"`: still running.  
   - `status = "done"`: completed, result is returned (including malicious score and model confidence).  
   - `status = "error"`: detection failed, error message is returned.

---

## Output Metrics

The GNN detection produces two core metrics:

- **Malicious Score (`score` / `score_pct`)**  
  - Interpreted as the probability that the contract is malicious.  
  - Computed as the average of the probabilities from all models:  
    `final_score = np.mean(all_probs)`  
  - `score_pct` is the percentage string representation (e.g. `"82.35%"`).

- **Model Confidence (`consistency` / `consistency_pct`)**  
  - Measures how consistent the different models are with each other.  
  - Defined as `1 - standard deviation` of the probabilities:  
    models closer together → smaller std → higher confidence.  

Example interpretations:

- **High malicious score + high confidence**: strongly suspected malicious contract (high risk).  
- **High malicious score + low confidence**: average looks malicious but models disagree; manual review strongly recommended.  
- **Low malicious score + high confidence**: models consistently consider the contract benign.  
- **Low malicious score + low confidence**: unstable conclusion; treat with caution and investigate further.


## License & Disclaimer

This project is intended **for research and educational purposes only** and does **not** constitute any form of investment advice or security guarantee.  
Always combine this tool with manual review and other security analysis methods when evaluating smart contracts.

