import os
import math
import torch
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


DATA_ROOT = "facts file convertor/dataset"
OUTPUT_DIR = "facts file convertor/processed_fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────── Opcode vocabulary (updated through Dencun) ────────────────────────
# Only affects feature dimensionality; does not change TSV parsing logic.
# Covers: Vandal abstract / classic EVM / added PUSH0, TLOAD/TSTORE, MCOPY, PREVRANDAO, CHAINID,
# SELFBALANCE, BASEFEE, BLOBHASH, BLOBBASEFEE, etc. [web:151][web:59][web:152][web:149][web:148][web:147][web:146]

OPCODE_VOCAB = [
    # Vandal abstract / TAC
    "CONST", "NOP", "LOG", "THROW", "THROWI",

    # Stack / Memory / Storage / Flow (including new opcodes)
    "STOP",
    "ADD","MUL","SUB","DIV","SDIV","MOD","SMOD","ADDMOD","MULMOD","EXP","SIGNEXTEND",
    "LT","GT","SLT","SGT","EQ","ISZERO",
    "AND","OR","XOR","NOT","BYTE",
    "SHL","SHR","SAR",                # Constantinople[web:146]

    "SHA3",

    "ADDRESS","BALANCE","ORIGIN","CALLER","CALLVALUE",
    "CALLDATALOAD","CALLDATASIZE","CALLDATACOPY",
    "CODESIZE","CODECOPY",
    "GASPRICE",
    "EXTCODESIZE","EXTCODECOPY","EXTCODEHASH",   # EXTCODEHASH added [web:146]
    "RETURNDATASIZE","RETURNDATACOPY",
    "BLOCKHASH",
    "COINBASE",
    "TIMESTAMP",
    "NUMBER",
    "DIFFICULTY","PREVRANDAO",        # 0x44 semantics changed to PREVRANDAO, keep DIFFICULTY name [web:149]
    "GASLIMIT",
    "CHAINID","SELFBALANCE",          # Istanbul[web:146]
    "BASEFEE",                        # London[web:146]
    "BLOBHASH","BLOBBASEFEE",         # Dencun EIP-4844/EIP-7516[web:147][web:148]

    "POP",
    "MLOAD","MSTORE","MSTORE8",
    "SLOAD","SSTORE",
    "TLOAD","TSTORE",                 # EIP-1153 [web:59]
    "MCOPY",                          # EIP-5656 [web:152]

    "JUMP","JUMPI","JUMPDEST",
    "PC","MSIZE","GAS",

    # PUSH series (most are converted to CONST in Vandal, but some paths may still keep the name)
    "PUSH0",
    "PUSH1","PUSH2","PUSH3","PUSH4","PUSH5","PUSH6","PUSH7",
    "PUSH8","PUSH9","PUSH10","PUSH11","PUSH12","PUSH13","PUSH14","PUSH15","PUSH16",
    "PUSH17","PUSH18","PUSH19","PUSH20","PUSH21","PUSH22","PUSH23","PUSH24",
    "PUSH25","PUSH26","PUSH27","PUSH28","PUSH29","PUSH30","PUSH31","PUSH32",

    # DUP / SWAP
    "DUP1","DUP2","DUP3","DUP4","DUP5","DUP6","DUP7","DUP8",
    "DUP9","DUP10","DUP11","DUP12","DUP13","DUP14","DUP15","DUP16",
    "SWAP1","SWAP2","SWAP3","SWAP4","SWAP5","SWAP6","SWAP7","SWAP8",
    "SWAP9","SWAP10","SWAP11","SWAP12","SWAP13","SWAP14","SWAP15","SWAP16",

    # Logging
    "LOG0","LOG1","LOG2","LOG3","LOG4",

    # System / Call / Create / Return
    "CREATE","CREATE2",
    "CALL","CALLCODE","DELEGATECALL","STATICCALL",
    "RETURN","REVERT","INVALID",
    "SELFDESTRUCT",
]

opcode2idx = {op: i for i, op in enumerate(OPCODE_VOCAB)}
UNK_OPCODE_IDX = len(OPCODE_VOCAB)   # Unknown opcodes are put in the last dimension

# Dangerous opcodes: keep the original set as primary, with slight additions (TSTORE/SELFDESTRUCT already included)
DANGEROUS_OPS = {
    "CALL","DELEGATECALL","STATICCALL","CALLCODE",
    "CREATE","CREATE2",
    "SSTORE","TSTORE",                # Transient storage writes are also sensitive [web:59]
    "SELFDESTRUCT",
    "JUMP","JUMPI",
    "ORIGIN","CALLER",
}

# ─────────────────────────── Utility functions ────────────────────────────

def normalize_hex(x):
    """Normalize various PC formats to a lowercase '0x...' string."""
    x = str(x).strip().lower()
    if x.startswith("0x"):
        return x
    if x.lstrip("0x").isdigit() or all(c in "0123456789abcdef" for c in x.lstrip("0x")):
        try:
            return hex(int(x, 16)) if x.startswith("0x") else hex(int(x))
        except ValueError:
            return x
    return x


def entropy_from_counts(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / total
    return float(-(probs * np.log(probs + 1e-9)).sum())


# ───────────────────────── Read facts files ───────────────────────

def read_block_structure(path):
    """
    block.facts format: <instruction_pc>\t<block_head_pc>
    Returns:
        instr2block: dict  pc_hex -> block_head_hex
        block_heads: list  sorted block heads (hex strings)
    """
    instr2block = {}
    block_heads_set = set()
    if not os.path.exists(path):
        return instr2block, []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pc   = normalize_hex(parts[0])
                head = normalize_hex(parts[1])
                instr2block[pc] = head
                block_heads_set.add(head)

    block_heads = sorted(block_heads_set, key=lambda x: int(x, 16))
    return instr2block, block_heads


def read_ops(path):
    """
    op.facts format: <pc>\t<opcode>
    Returns dict: pc_hex -> [opcode, ...]
    """
    ops = defaultdict(list)
    if not os.path.exists(path):
        return ops
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pc = normalize_hex(parts[0])
                op = parts[1].strip().upper()
                ops[pc].append(op)
    return ops


def read_entry_exit(entry_path, exit_path, instr2block):
    entries, exits = set(), set()
    for path, target_set in [(entry_path, entries), (exit_path, exits)]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    pc = normalize_hex(line.strip())
                    if pc and pc in instr2block:
                        target_set.add(instr2block[pc])
    return entries, exits


def read_def_locations(path):
    """def.facts format: <var>\t<pc>"""
    defs = {}
    if not os.path.exists(path):
        return defs
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                defs[parts[0]] = normalize_hex(parts[1])
    return defs


# ─────────────────────── Contract graph construction ──────────────────────────────

def convert_contract(fact_dir, label):
    # 1. Read block structure
    instr2block, block_heads = read_block_structure(
        os.path.join(fact_dir, "block.facts")
    )
    if not block_heads:
        return None

    block_id  = {b: i for i, b in enumerate(block_heads)}
    num_nodes = len(block_heads)

    # 2. Read auxiliary information
    op_map   = read_ops(os.path.join(fact_dir, "op.facts"))
    entries, exits = read_entry_exit(
        os.path.join(fact_dir, "entry.facts"),
        os.path.join(fact_dir, "exit.facts"),
        instr2block,
    )
    var_defs = read_def_locations(os.path.join(fact_dir, "def.facts"))

    # 3. Aggregate instructions by block
    block_instrs = defaultdict(list)
    for pc, blk in instr2block.items():
        block_instrs[blk].append(pc)

    # 4. Build node features
    X = []
    VOCAB_SIZE = len(OPCODE_VOCAB) + 1  # +1 for UNK

    for blk in block_heads:
        pcs = block_instrs[blk]
        ops = []
        for pc in pcs:
            ops.extend(op_map.get(pc, []))

        counter = Counter(ops)

        # (a) Opcode frequency (log1p): VOCAB_SIZE dimensions
        feat = [0.0] * VOCAB_SIZE
        for op, c in counter.items():
            feat[opcode2idx.get(op, UNK_OPCODE_IDX)] += math.log1p(c)

        # (b) Danger opcode log1p counts: len(DANGEROUS_OPS) dimensions
        danger_count = 0
        for d in sorted(DANGEROUS_OPS):   # Fixed order to preserve feature dimension consistency
            dc = counter[d]
            feat.append(math.log1p(dc))
            danger_count += dc

        # (c) Block-level statistical features: 5 dimensions
        feat.append(math.log1p(len(ops)))               # total instructions (log)
        feat.append(entropy_from_counts(counter))       # opcode entropy
        feat.append(danger_count / max(1, len(ops)))    # dangerous opcode ratio
        feat.append(1.0 if blk in entries else 0.0)     # is entry block
        feat.append(1.0 if blk in exits   else 0.0)     # is exit block

        X.append(feat)

    # 5. Build edges
    edges    = []
    edge_set = set()

    # ── 5a. CFGEdge.facts (control flow edges)──────────────────────────────
    cfg_path = os.path.join(fact_dir, "CFGEdge.facts")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        src_pc = hex(int(parts[0]))   # decimal -> hex
                        tgt_pc = hex(int(parts[1]))
                        ub = instr2block.get(src_pc)
                        vb = instr2block.get(tgt_pc)
                        if ub and vb and ub in block_id and vb in block_id:
                            s, t = block_id[ub], block_id[vb]
                            key = (s, t)
                            if key not in edge_set:
                                edges.append([s, t])
                                edge_set.add(key)
                    except (ValueError, KeyError):
                        continue

    # ── 5b. edge.facts (fallback)─────────────────────────────────
    if not edges:
        edge_path = os.path.join(fact_dir, "edge.facts")
        if os.path.exists(edge_path):
            with open(edge_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            u  = normalize_hex(parts[0])
                            v  = normalize_hex(parts[1])
                            ub = instr2block.get(u)
                            vb = instr2block.get(v)
                            if ub and vb and ub in block_id and vb in block_id:
                                s, t = block_id[ub], block_id[vb]
                                key = (s, t)
                                if key not in edge_set:
                                    edges.append([s, t])
                                    edge_set.add(key)
                        except (ValueError, KeyError):
                            continue

    # ── 5c. use.facts (data flow edges)─────────────────────────────────
    use_path = os.path.join(fact_dir, "use.facts")
    if os.path.exists(use_path):
        with open(use_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:   # ignore the 3rd column operand index
                    try:
                        var    = parts[0]
                        use_pc = normalize_hex(parts[1])
                        def_pc = var_defs.get(var)
                        if not def_pc:
                            continue
                        ub = instr2block.get(def_pc)
                        vb = instr2block.get(use_pc)
                        if ub and vb and ub != vb and ub in block_id and vb in block_id:
                            s, t = block_id[ub], block_id[vb]
                            key = (s, t)
                            if key not in edge_set:
                                edges.append([s, t])
                                edge_set.add(key)
                    except (KeyError, ValueError):
                        continue

    # 6. Assemble Data object
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([label], dtype=torch.long),
        num_nodes  = num_nodes,
    )
    data.has_edges  = edge_index.size(1) > 0
    data.num_edges  = edge_index.size(1)
    return data


# ─────────────────────── Global feature normalization ─────────────────────────

def normalize_features(graphs):
    """
    Apply per-feature StandardScaler to node features of all graphs.
    """
    all_x = torch.cat([g.x for g in graphs], dim=0).numpy()
    scaler = StandardScaler()
    scaler.fit(all_x)

    for g in graphs:
        g.x = torch.tensor(
            scaler.transform(g.x.numpy()), dtype=torch.float
        )
    return graphs, scaler


# ──────────────────────── Main process ────────────────────────────────

def process_dataset():
    graphs = []
    stats  = defaultdict(int)

    for cls, label in [("benign", 0), ("malicious", 1)]:
        base = os.path.join(DATA_ROOT, cls)
        if not os.path.exists(base):
            print(f"[!] Directory not found: {base}")
            continue

        names = sorted(os.listdir(base))
        for name in names:
            contract_path = os.path.join(base, name)
            if not os.path.isdir(contract_path):
                continue

            stats["total"] += 1
            try:
                g = convert_contract(contract_path, label)
                if g is not None:
                    graphs.append(g)
                    stats["success"]    += 1
                    stats["with_edges"] += int(g.has_edges)
                else:
                    stats["failed"] += 1
                    print(f"[!] Skipped (no blocks): {contract_path}")
            except Exception as e:
                stats["failed"] += 1
                print(f"[!] Processing error {contract_path}: {e}")

    if not graphs:
        print("[✗] No contracts processed successfully, exiting.")
        return

    # Feature normalization
    print(f"\n[*] Normalizing node features for {len(graphs)} graphs...")
    graphs, scaler = normalize_features(graphs)

    # Save
    out_pt      = os.path.join(OUTPUT_DIR, "graphs_fusion_final.pt")
    out_scaler  = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
    torch.save(graphs, out_pt)

    import pickle
    with open(out_scaler, "wb") as f:
        pickle.dump(scaler, f)

    num_features = graphs[0].num_node_features
    print("\n=== Final statistics ===")
    print(f"Total:         {stats['total']}")
    print(f"Success:       {stats['success']}")
    print(f"Failed/Skipped:{stats['failed']}")
    print(f"Graphs with edges: {stats['with_edges']}")
    print(f"Node feature dim: {num_features}")
    print(f"Average nodes:  {sum(g.num_nodes for g in graphs)/len(graphs):.1f}")
    print(f"Average edges:  {sum(g.num_edges for g in graphs)/max(1, stats['with_edges']):.1f}")
    print(f"Saved graph data: {out_pt}")
    print(f"Saved scaler: {out_scaler}")


if __name__ == "__main__":
    process_dataset()