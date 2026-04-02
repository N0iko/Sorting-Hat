"""
Smart contract detection pipeline:
1. Supports single address or batch detection (read contract addresses from a txt file)
2. Download bytecode and save as .hex
3. Use vandal-master to convert .hex to CFG facts
4. Use GNN/detect.py (with scaler normalization) for malicious contract detection
5. Save results as txt report + CSV (for later analysis)
"""

import os
import sys
import re
import csv
import importlib
import subprocess
import time
from datetime import datetime


# ===== Configuration (modify as needed) =====
FYP_ROOT = os.path.dirname(os.path.abspath(__file__))
VANDAL_ROOT = os.path.join(FYP_ROOT, "vandal-master")
HEX_INPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_input_hex")
CFG_OUTPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_output_cfg")
GNN_DIR = os.path.join(FYP_ROOT, "GNN")
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/NKKDehg7-0Cj4b3FUJugH"
BATCH_RESULTS_DIR = os.path.join(FYP_ROOT, "batch_results")
DELAY_BETWEEN_ADDRESSES = 0.5   # Delay between processing each address (seconds)
# ============================

# Ethereum address regex
_ETH_ADDR_RE = re.compile(r'^0x[0-9a-fA-F]{40}$')


# ──────────────────────── Utility functions ───────────────────────────────

def normalize_address(address: str):
    """Normalize and validate the address format. Return None for invalid addresses."""
    address = address.strip()
    if not address:
        return None
    if not address.startswith("0x"):
        address = "0x" + address
    if not _ETH_ADDR_RE.match(address):
        print(f"[!] Invalid address format, skipped: {address}")
        return None
    return address


def get_bytecode(address: str):
    """Fetch contract bytecode from the chain and return a hex string."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    try:
        code = w3.eth.get_code(Web3.to_checksum_address(address)).hex()
        return None if code == "0x" else code
    except Exception as e:
        print(f"[!] Failed to fetch bytecode: {e}")
        return None


def save_hex(address: str, code: str):
    """Save bytecode as a .hex file using the checksum address as filename."""
    from web3 import Web3
    name = Web3.to_checksum_address(address)
    os.makedirs(HEX_INPUT_DIR, exist_ok=True)
    path = os.path.join(HEX_INPUT_DIR, f"{name}.hex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[✓] Saved {name}.hex -> {path}")
    return path, name


def clean_and_create_dir(dir_path):
    """Clean and recreate the directory to ensure fresh output for each decompile."""
    import shutil
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def run_vandal_decompile(hex_path, output_name):
    """
    Smart two-stage decompilation:
    1. First stage: strict alignment with training parameters (iterations=5), timeout 200s.
    2. Second stage: if timed out, automatically fall back to decompile_nodataflow.
    """
    out_dir = os.path.join(CFG_OUTPUT_DIR, output_name)
    decompile_script = os.path.join(VANDAL_ROOT, "bin", "decompile")
    
    # The previous error was caused by the missing function definition for the line below
    clean_and_create_dir(out_dir)
    
    # Core alignment parameters
    cmd = [
        sys.executable, decompile_script, hex_path, 
        "-t", out_dir, 
        "-n",
        "-c", "bailout_seconds=10,max_iterations=5,widen_threshold=3,analytics=false,extract_functions=false"
    ]
    
    print(f"[*] Running Vandal decompilation (alignment mode: iterations=5)...")
    
    try:
        # Try standard alignment decompilation
        res = subprocess.run(cmd, cwd=VANDAL_ROOT, capture_output=True, text=True, timeout=200)
        
        if res.returncode == 0:
            print(f"[✓] Decompilation succeeded.")
            return out_dir
        else:
            print(f"[X] Vandal error: {res.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print("[X] Decompilation timed out (200s). Falling back to decompile_nodataflow (skip dataflow analysis)...")
        
        fallback_script = os.path.join(VANDAL_ROOT, "bin", "decompile_nodataflow")
        fallback_cmd = [sys.executable, fallback_script, hex_path, "-t", out_dir, "-n"]
        
        try:
            fallback_res = subprocess.run(fallback_cmd, cwd=VANDAL_ROOT, capture_output=True, text=True, timeout=120)
            if fallback_res.returncode == 0:
                print(f"[✓] (fallback mode) Decompilation succeeded.")
                return out_dir
            else:
                print(f"[X] Fallback decompilation error: {fallback_res.stderr[:200]}")
                return None
        except Exception as e:
            print(f"[X] Fallback runtime error: {e}")
            return None
            
    except Exception as e:
        print(f"[X] Runtime error: {e}")

    return None


def run_gnn_detect(facts_dir: str, return_result: bool = False):
    """
    Call GNN/detect.py to detect on the specified facts directory.
    Use importlib.reload to ensure each batch run is executed fresh,
    avoiding stale scaler or model path caching.
    """
    if not os.path.isdir(GNN_DIR):
        print(f"[!] GNN directory not found: {GNN_DIR}")
        return None

    if GNN_DIR not in sys.path:
        sys.path.insert(0, GNN_DIR)

    orig_cwd = os.getcwd()
    try:
        os.chdir(GNN_DIR)
        import detect
        importlib.reload(detect)   # Important: reload each time to prevent stale state

        if return_result:
            return detect.run_detection_result(target_facts_dir=facts_dir)
        detect.run_detection(target_facts_dir=facts_dir)
    finally:
        os.chdir(orig_cwd)

    return None


# ──────────────────────── Address reading ───────────────────────────────

def read_addresses_from_file(file_path: str):
    """
    Read a list of contract addresses from a txt file.
    Supports:
    - one address per line
    - 'address label' format per line
    - automatically skip blank lines, # comment lines, duplicate addresses, invalid addresses
    """
    addresses = []
    seen      = set()
    pattern   = re.compile(r'0x[0-9a-fA-F]{40}')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = pattern.search(line)
                if not match:
                    print(f"[!] No valid address on line {line_no}, skipped: {line}")
                    continue
                addr = match.group(0)
                key  = addr.lower()
                if key in seen:
                    print(f"[-] Duplicate address on line {line_no}, skipped: {addr}")
                    continue
                seen.add(key)
                addresses.append(addr)
        return addresses
    except Exception as e:
        print(f"[!] Failed to read address file: {e}")
        return []


# ──────────────────────── Result saving ───────────────────────────────

def save_batch_results(results: list, timestamp: str):
    """
    Save batch detection results:
    - batch_results_<timestamp>.txt readable report
    - batch_results_<timestamp>.csv structured data (for later analysis)
    """
    os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
    base_name = f"batch_results_{timestamp}"

    # ── TXT report ──
    txt_path = os.path.join(BATCH_RESULTS_DIR, f"{base_name}.txt")
    success_count = sum(1 for r in results if r['success'])
    fail_count    = len(results) - success_count

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Batch detection report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total: {len(results)}  Success: {success_count}  Fail: {fail_count}\n\n")
        f.write("-" * 80 + "\n")

        for idx, r in enumerate(results, 1):
            f.write(f"[{idx}] Address: {r['address']}\n")
            if r['success']:
                det = r['detection']
                f.write(f"    Status   : ✓ Success\n")
                f.write(f"    Conclusion: {det.get('conclusion', 'N/A')}\n")
                f.write(f"    Malicious score: {det.get('score_pct', 'N/A')}  "
                        f"(raw: {det.get('score', 'N/A'):.6f})\n")
                f.write(f"    Consistency: {det.get('consistency_pct', 'N/A')}\n")
                f.write(f"    Models used: {det.get('models_used', 'N/A')}\n")
                f.write(f"    Time    : {r['time']:.2f} sec\n")
            else:
                f.write(f"    Status: ✗ Failed\n")
                f.write(f"    Error: {r['error']}\n")
                f.write(f"    Time: {r['time']:.2f} sec\n")
            f.write("\n")

    print(f"[✓] TXT report saved: {txt_path}")

    # ── CSV (structured) ──
    csv_path = os.path.join(BATCH_RESULTS_DIR, f"{base_name}.csv")
    fieldnames = [
        "index", "address", "success", "error",
        "conclusion", "conclusion_key",
        "score", "score_pct",
        "consistency", "consistency_pct",
        "models_used", "time_sec",
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(results, 1):
            det = r.get('detection') or {}
            writer.writerow({
                "index"           : idx,
                "address"         : r['address'],
                "success"         : r['success'],
                "error"           : r.get('error', ''),
                "conclusion"      : det.get('conclusion', ''),
                "conclusion_key"  : det.get('conclusion_key', ''),
                "score"           : det.get('score', ''),
                "score_pct"       : det.get('score_pct', ''),
                "consistency"     : det.get('consistency', ''),
                "consistency_pct" : det.get('consistency_pct', ''),
                "models_used"     : det.get('models_used', ''),
                "time_sec"        : f"{r['time']:.2f}",
            })

    print(f"[✓] CSV saved: {csv_path}")
    return txt_path, csv_path


# ──────────────────────── Core pipeline ───────────────────────────────

def run_pipeline(address: str):
    """
    Run the full detection pipeline for a single address.
    Returns: (success: bool, detection_dict | error_str)
    """
    address = normalize_address(address)
    if not address:
        return False, "Invalid address format"

    code = get_bytecode(address)
    if not code:
        return False, "Failed to fetch bytecode. Check the address and RPC"

    hex_path, output_name = save_hex(address, code)

    facts_dir = run_vandal_decompile(hex_path, output_name)
    if not facts_dir or not os.path.isdir(facts_dir):
            return False, "CFG generation failed (Vandal decompilation unsuccessful)"

    result = run_gnn_detect(facts_dir, return_result=True)
    if result is None:
        return False, "GNN detection failed"
    return True, result


# ──────────────────────── Batch detection ───────────────────────────────

def run_batch_detection(file_path: str,
                        continue_on_error: bool = True,
                        save_results: bool = True):
    """Batch detect contract addresses from a txt file."""
    print("\n" + "=" * 65)
    print("  Batch smart contract detection mode")
    print("=" * 65)

    addresses = read_addresses_from_file(file_path)
    if not addresses:
        print("[✗] No valid contract addresses were read")
        return None, None

    print(f"[*] Read {len(addresses)} valid addresses, starting detection...\n")

    results       = []
    success_count = 0
    fail_count    = 0

    for idx, address in enumerate(addresses, 1):
        print(f"\n{'=' * 65}")
        print(f"  [{idx}/{len(addresses)}] Detecting address: {address}")
        print(f"{'=' * 65}")

        t0 = time.time()
        try:
            success, data = run_pipeline(address)
            elapsed = time.time() - t0

            if success:
                success_count += 1
                det = data  # data is the detection dict
                print(f"[✓] Detection succeeded -> {det.get('conclusion')}  "
                      f"score={det.get('score', 0):.4f}  "
                      f"({elapsed:.2f}s)")
                results.append({
                    "address"  : address,
                    "success"  : True,
                    "detection": det,
                    "time"     : elapsed,
                    "error"    : None,
                })
            else:
                fail_count += 1
                print(f"[✗] Detection failed: {data}  ({elapsed:.2f}s)")
                results.append({
                    "address"  : address,
                    "success"  : False,
                    "detection": None,
                    "time"     : elapsed,
                    "error"    : str(data),
                })
                if not continue_on_error:
                    print("[!] continue_on_error=False, stopping batch detection")
                    break

        except Exception as e:
            elapsed = time.time() - t0
            fail_count += 1
            print(f"[✗] Execution error: {e}  ({elapsed:.2f}s)")
            results.append({
                "address"  : address,
                "success"  : False,
                "detection": None,
                "time"     : elapsed,
                "error"    : f"Error: {e}",
            })
            if not continue_on_error:
                print("[!] continue_on_error=False, stop batch detection")
                break

        if idx < len(addresses):
            time.sleep(DELAY_BETWEEN_ADDRESSES)

    # Summary
    print("\n" + "=" * 65)
    print("  Batch detection complete")
    print("=" * 65)
    print(f"  Total: {len(results)}  Success: {success_count}  Fail: {fail_count}")
    if results:
        print(f"  Success rate: {success_count / len(results) * 100:.1f}%")

    txt_path = csv_path = None
    if save_results and results:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_path, csv_path = save_batch_results(results, ts)

    return results, (txt_path, csv_path)


# ──────────────────────── Main entry ─────────────────────────────────

def main():
    print("=" * 65)
    print("  Smart contract detection pipeline (single address / batch mode)")
    print("=" * 65)
    print("\nPlease select detection mode:")
    print("  1. Single address detection")
    print("  2. Batch detection (read addresses list from txt file)")
    print("  3. Exit")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == '1':
        address = input("\nEnter smart contract address (including 0x): ").strip()
        if not address:
            print("[!] No address entered, exiting")
            return
        success, data = run_pipeline(address)
        print("\n" + "=" * 50)
        if success:
            det = data
            print(f"  Address  : {address}")
            print(f"  Conclusion: {det.get('conclusion')}")
            print(f"  Malicious score: {det.get('score_pct')}  (raw: {det.get('score', 0):.6f})")
            print(f"  Consistency: {det.get('consistency_pct')}")
            print(f"  Models used: {det.get('models_used')}")
        else:
            print(f"  [✗] Detection failed: {data}")
        print("=" * 50)

    elif choice == '2':
        file_path = input("\nEnter path to txt file containing contract addresses: ").strip()
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
            return

        cont  = input("Continue on error? (y/n, default y): ").strip().lower()
        save  = input("Save detection results? (y/n, default y): ").strip().lower()
        run_batch_detection(
            file_path,
            continue_on_error=(cont != 'n'),
            save_results=(save != 'n'),
        )

    elif choice == '3':
        print("Exiting program")

    else:
        print("[!] Invalid choice")


if __name__ == "__main__":
    main()