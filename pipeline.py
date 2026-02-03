# -*- coding: utf-8 -*-
"""
Malicious Smart Contract Detection Pipeline:
1. User inputs the contract address
2. Download the bytecode and save it as .hex 
3. Use vandal-master to convert .hex to CFG facts
4. Use GNN/detect.py to perform malicious contract detection and output the result
"""

import os
import sys
import subprocess


FYP_ROOT = os.getcwd()
VANDAL_ROOT = os.path.join(FYP_ROOT, "vandal-master")
HEX_INPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_input_hex")
CFG_OUTPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_output_cfg")
GNN_DIR = os.path.join(FYP_ROOT, "GNN")
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/NKKDehg7-0Cj4b3FUJugH"



def get_bytecode(address):
    """Get the bytecode from the blockchain and return it as a hexadecimal string."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    try:
        code = w3.eth.get_code(Web3.to_checksum_address(address)).hex()
        return None if code == "0x" else code
    except Exception as e:
        print(f"[!] Failed to get bytecode: {e}")
        return None


def save_hex(address, code):
    """Save the bytecode as a .hex file, using the checksum address as the filename."""
    from web3 import Web3
    name = Web3.to_checksum_address(address)
    os.makedirs(HEX_INPUT_DIR, exist_ok=True)
    path = os.path.join(HEX_INPUT_DIR, f"{name}.hex")
    with open(path, "w") as f:
        f.write(code)
    print(f"[✓] Saved {name}.hex -> {path}")
    return path, name


def run_vandal_decompile(hex_path, output_name):
    """Call vandal-master's bin/decompile to convert .hex to CFG facts."""
    os.makedirs(CFG_OUTPUT_DIR, exist_ok=True)
    out_dir = os.path.join(CFG_OUTPUT_DIR, output_name)
    decompile_script = os.path.join(VANDAL_ROOT, "bin", "decompile")
    cmd = [sys.executable, decompile_script, hex_path, "-t", out_dir, "-n"]
    print(f"[*] Running Vandal decompile: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=VANDAL_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"[!] Vandal returned an error: {result.stderr or result.stdout}")
            return None
        print(f"[✓] CFG has been output to {out_dir}")
        return out_dir
    except subprocess.TimeoutExpired:
        print("[!] Vandal execution timed out")
        return None
    except Exception as e:
        print(f"[!] Failed to execute Vandal: {e}")
        return None


def run_gnn_detect(facts_dir, return_result=False):
    """Call GNN/detect.py to perform detection on the specified facts directory. If return_result=True, return the result dictionary."""
    if not os.path.isdir(GNN_DIR):
        print(f"[!] Could not find GNN directory: {GNN_DIR}")
        return None if return_result else None
    if GNN_DIR not in sys.path:
        sys.path.insert(0, GNN_DIR)
    orig_cwd = os.getcwd()
    try:
        os.chdir(GNN_DIR)
        import detect
        if return_result:
            return detect.run_detection_result(target_facts_dir=facts_dir)
        detect.run_detection(target_facts_dir=facts_dir)
    finally:
        os.chdir(orig_cwd)
    return None


def main():
    print("=" * 50)
    print("  Malicious Smart Contract Detection Pipeline (Address -> Bytecode -> CFG -> Detection)")
    print("=" * 50)

    address = input("\nPlease enter the contract address (including 0x): ").strip()
    if not address:
        print("[!] No address entered, exiting")
        return
    if not address.startswith("0x"):
        address = "0x" + address


    print("\n[Step 1] Downloading contract bytecode...")
    code = get_bytecode(address)
    if not code:
        print("[✗] Failed to get the bytecode for the address, please check the address and RPC")
        return
    hex_path, output_name = save_hex(address, code)

  
    print("\n[Step 2] Vandal decompile to generate CFG facts...")
    facts_dir = run_vandal_decompile(hex_path, output_name)
    if not facts_dir or not os.path.isdir(facts_dir):
        print("[✗] Failed to generate CFG, pipeline terminated")
        return

  
    print("\n[Step 3] GNN malicious contract detection...")
    run_gnn_detect(facts_dir)
    print("\nPipeline completed.")


def run_pipeline(address):

    address = address.strip()
    if not address:
        return False, "No address entered"
    if not address.startswith("0x"):
        address = "0x" + address

    code = get_bytecode(address)
    if not code:
        return False, "Failed to get the bytecode for the address, please check the address and RPC"
    hex_path, output_name = save_hex(address, code)

    facts_dir = run_vandal_decompile(hex_path, output_name)
    if not facts_dir or not os.path.isdir(facts_dir):
        return False, "Failed to generate CFG (Vandal decompile failed)"

    result = run_gnn_detect(facts_dir, return_result=True)
    if result is None:
        return False, "GNN detection failed"
    return True, result


if __name__ == "__main__":
    main()
