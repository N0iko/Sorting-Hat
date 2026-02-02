import subprocess
import pathlib

# 每個合約的最長允許時間（秒）
TIMEOUT = 120

HEX_DIR = pathlib.Path("contracts_input_hex")
OUT_ROOT = pathlib.Path("contracts_output_cfg")
OUT_ROOT.mkdir(exist_ok=True)

for hex_file in HEX_DIR.glob("*.hex"):
    name = hex_file.stem
    out_dir = OUT_ROOT / name
    print(f"[*] Processing {name}", flush=True)

    cmd = [
        "python", "bin/decompile",
        str(hex_file),
        "-t", str(out_dir),
        "-n",
    ]

    try:
        subprocess.run(cmd, check=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"[!] Timeout ({TIMEOUT}s), skip {name}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error on {name}: {e}", flush=True)