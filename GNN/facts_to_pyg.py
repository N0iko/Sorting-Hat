import os
import math
import torch
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data

DATA_ROOT = "facts file convertor/dataset"
OUTPUT_DIR = "facts file convertor/processed_fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPCODE_VOCAB = [
    "PUSH1","PUSH2","PUSH4","PUSH32","DUP1","DUP2","SWAP1","SWAP2",
    "POP","MLOAD","MSTORE","MSTORE8",
    "JUMP","JUMPI","JUMPDEST",
    "ISZERO","EQ","LT","GT",
    "ADD","SUB","MUL","DIV","EXP",
    "AND","OR","XOR","NOT","BYTE","SHL","SHR","SAR",
    "PC","MSIZE","GAS",
    "ADDRESS","BALANCE","ORIGIN","CALLER","CALLVALUE",
    "CODESIZE","CODECOPY","RETURNDATASIZE","RETURNDATACOPY",
    "BLOCKHASH","COINBASE","TIMESTAMP","NUMBER","DIFFICULTY","GASLIMIT",
    "SLOAD","SSTORE",
    "CALL","STATICCALL","DELEGATECALL","CALLCODE",
    "CREATE","CREATE2",
    "EXTCODESIZE","EXTCODECOPY",
    "SELFDESTRUCT",
    "RETURN","REVERT","INVALID","STOP",
    "LOG0","LOG1","LOG2","LOG3","LOG4"
]

opcode2idx = {op: i for i, op in enumerate(OPCODE_VOCAB)}
UNK_OPCODE_IDX = len(OPCODE_VOCAB)

DANGEROUS_OPS = {
    "CALL","DELEGATECALL","STATICCALL","CALLCODE",
    "CREATE","CREATE2",
    "SSTORE","SELFDESTRUCT",
    "JUMP","JUMPI","ORIGIN","CALLER"
}


def normalize_hex(x):
    x = str(x).strip().lower()
    if x.startswith("0x"):
        return x
    if x.isdigit():
        return hex(int(x))
    return x

def entropy_from_counts(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counter.values())) / total
    return float(-(probs * np.log(probs + 1e-9)).sum())


def read_block_structure(path):
    instr2block = {}
    block_heads = set()
    if not os.path.exists(path):
        return instr2block, []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pc, head = parts[0], parts[1]
                pc, head = normalize_hex(pc), normalize_hex(head)
                instr2block[pc] = head
                block_heads.add(head)

    return instr2block, sorted(block_heads, key=lambda x: int(x, 16))

def read_ops(path):
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
                pc, op = parts[0], parts[1]
                ops[normalize_hex(pc)].append(op.strip().upper())
    return ops

def read_entry_exit(entry_path, exit_path, instr2block):
    entries, exits = set(), set()
    if os.path.exists(entry_path):
        with open(entry_path, 'r', encoding='utf-8') as f:
            for l in f:
                pc = normalize_hex(l.strip())
                if pc and pc in instr2block:
                    entries.add(instr2block[pc])
    if os.path.exists(exit_path):
        with open(exit_path, 'r', encoding='utf-8') as f:
            for l in f:
                pc = normalize_hex(l.strip())
                if pc and pc in instr2block:
                    exits.add(instr2block[pc])
    return entries, exits

def read_def_locations(path):
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
                var, pc = parts[0], parts[1]
                defs[var] = normalize_hex(pc)
    return defs


def convert_contract(fact_dir, label):
    instr2block, block_heads = read_block_structure(os.path.join(fact_dir, "block.facts"))
    if not block_heads:
        return None

    block_id = {b: i for i, b in enumerate(block_heads)}
    num_nodes = len(block_heads)


    op_map = read_ops(os.path.join(fact_dir, "op.facts"))
    entries, exits = read_entry_exit(
        os.path.join(fact_dir, "entry.facts"),
        os.path.join(fact_dir, "exit.facts"),
        instr2block
    )
    var_defs = read_def_locations(os.path.join(fact_dir, "def.facts"))


    block_instrs = defaultdict(list)
    for pc, blk in instr2block.items():
        block_instrs[blk].append(pc)


    X = []
    for blk in block_heads:
        pcs = block_instrs[blk]
        ops = []
        for pc in pcs:
            ops.extend(op_map.get(pc, []))

        counter = Counter(ops)
        feat = [0.0] * (len(OPCODE_VOCAB) + 1)

 
        for op, c in counter.items():
            feat[opcode2idx.get(op, UNK_OPCODE_IDX)] += math.log1p(c)


        danger_count = 0
        for d in DANGEROUS_OPS:
            dc = counter[d]
            feat.append(math.log1p(dc))
            danger_count += dc


        feat.append(len(ops)) 
        feat.append(entropy_from_counts(counter))  
        feat.append(danger_count / max(1, len(ops)))  
        feat.append(1.0 if blk in entries else 0.0) 
        feat.append(1.0 if blk in exits else 0.0)  
        feat.append(block_id[blk] / num_nodes)  

        X.append(feat)

  
    edges = []
    edge_set = set()  # 用於去重

 
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
                        s, t = int(parts[0]), int(parts[1])
                
                        if 0 <= s < num_nodes and 0 <= t < num_nodes:
                            edge_key = (s, t)
                            if edge_key not in edge_set:
                                edges.append([s, t])
                                edge_set.add(edge_key)
                    except (ValueError, IndexError):
                        continue
    else:
        
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
                            u, v = map(normalize_hex, parts[:2])
                            ub, vb = instr2block.get(u), instr2block.get(v)
                            if ub and vb and ub in block_id and vb in block_id:
                                s, t = block_id[ub], block_id[vb]
                                edge_key = (s, t)
                                if edge_key not in edge_set:
                                    edges.append([s, t])
                                    edge_set.add(edge_key)
                        except (ValueError, KeyError):
                            continue

    
    use_path = os.path.join(fact_dir, "use.facts")
    if os.path.exists(use_path):
        with open(use_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        var, use_pc = parts[0], normalize_hex(parts[1])
                        def_pc = var_defs.get(var)
                        if not def_pc:
                            continue
                        ub, vb = instr2block.get(def_pc), instr2block.get(use_pc)
                        if ub and vb and ub != vb and ub in block_id and vb in block_id:
                            s, t = block_id[ub], block_id[vb]
                            edge_key = (s, t)
                            if edge_key not in edge_set:
                                edges.append([s, t])
                                edge_set.add(edge_key)
                    except (KeyError, ValueError):
                        continue

    
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes
    )
    data.has_edges = edge_index.size(1) > 0
    data.num_edges = edge_index.size(1)
    return data

def process_dataset():
    
    graphs = []
    stats = defaultdict(int)

    for cls, label in [("benign", 0), ("malicious", 1)]:
        base = os.path.join(DATA_ROOT, cls)
        if not os.path.exists(base):
            print(f"Warning: Directory {base} does not exist")
            continue
        
        for name in os.listdir(base):
            contract_path = os.path.join(base, name)
            if not os.path.isdir(contract_path):
                continue
                
            stats["total"] += 1
            try:
                g = convert_contract(contract_path, label)
                if g:
                    graphs.append(g)
                    stats["success"] += 1
                    stats["with_edges"] += int(g.has_edges)
                else:
                    stats["failed"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"Error processing {contract_path}: {e}")

    if not graphs:
        print("Error: No graphs were successfully processed!")
        return

    out = os.path.join(OUTPUT_DIR, "graphs_fusion_final.pt")
    torch.save(graphs, out)

    print("\n=== Final Statistics ===")
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"With edges: {stats['with_edges']}")
    print(f"Avg nodes: {sum(g.num_nodes for g in graphs)/len(graphs):.1f}")
    print(f"Avg edges: {sum(g.num_edges for g in graphs)/max(1,stats['with_edges']):.1f}")
    print(f"Saved to {out}")

if __name__ == "__main__":
    process_dataset()
