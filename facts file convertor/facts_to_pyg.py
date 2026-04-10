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



OPCODE_VOCAB = [
    "CONST", "NOP", "LOG", "THROW", "THROWI", "STOP",
    "ADD","MUL","SUB","DIV","SDIV","MOD","SMOD","ADDMOD","MULMOD","EXP","SIGNEXTEND",
    "LT","GT","SLT","SGT","EQ","ISZERO",
    "AND","OR","XOR","NOT","BYTE",
    "SHL","SHR","SAR", "SHA3",
    "ADDRESS","BALANCE","ORIGIN","CALLER","CALLVALUE",
    "CALLDATALOAD","CALLDATASIZE","CALLDATACOPY",
    "CODESIZE","CODECOPY", "GASPRICE",
    "EXTCODESIZE","EXTCODECOPY","EXTCODEHASH", 
    "RETURNDATASIZE","RETURNDATACOPY",
    "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER",
    "DIFFICULTY","PREVRANDAO", "GASLIMIT",
    "CHAINID","SELFBALANCE", "BASEFEE", 
    "BLOBHASH","BLOBBASEFEE",
    "POP", "MLOAD","MSTORE","MSTORE8",
    "SLOAD","SSTORE", "TLOAD","TSTORE",  
    "MCOPY",
    "JUMP","JUMPI","JUMPDEST",
    "PC","MSIZE","GAS",
    "PUSH0",
    "PUSH1","PUSH2","PUSH3","PUSH4","PUSH5","PUSH6","PUSH7",
    "PUSH8","PUSH9","PUSH10","PUSH11","PUSH12","PUSH13","PUSH14","PUSH15","PUSH16",
    "PUSH17","PUSH18","PUSH19","PUSH20","PUSH21","PUSH22","PUSH23","PUSH24",
    "PUSH25","PUSH26","PUSH27","PUSH28","PUSH29","PUSH30","PUSH31","PUSH32",
    "DUP1","DUP2","DUP3","DUP4","DUP5","DUP6","DUP7","DUP8",
    "DUP9","DUP10","DUP11","DUP12","DUP13","DUP14","DUP15","DUP16",
    "SWAP1","SWAP2","SWAP3","SWAP4","SWAP5","SWAP6","SWAP7","SWAP8",
    "SWAP9","SWAP10","SWAP11","SWAP12","SWAP13","SWAP14","SWAP15","SWAP16",
    "LOG0","LOG1","LOG2","LOG3","LOG4",
    "CREATE","CREATE2",
    "CALL","CALLCODE","DELEGATECALL","STATICCALL",
    "RETURN","REVERT","INVALID",
    "SELFDESTRUCT",
]

opcode2idx = {op: i for i, op in enumerate(OPCODE_VOCAB)}
UNK_OPCODE_IDX = len(OPCODE_VOCAB)

DANGEROUS_OPS = {
    "CALL","DELEGATECALL","STATICCALL","CALLCODE",
    "CREATE","CREATE2",
    "SSTORE","TSTORE",
    "SELFDESTRUCT",
    "JUMP","JUMPI","ORIGIN","CALLER",
}

CRITICAL_WEIGHTS = {
    "DELEGATECALL": 5.0,
    "ORIGIN": 5.0,
    "SELFDESTRUCT": 4.0,
    "CREATE": 3.0,
    "CREATE2": 3.0,
    "TLOAD": 3.0,
    "TSTORE": 3.0
}



def normalize_hex(x):
    x = str(x).strip().lower()
    if x.startswith("0x"): return x
    if x.lstrip("0x").isdigit() or all(c in "0123456789abcdef" for c in x.lstrip("0x")):
        try:
            return hex(int(x, 16)) if x.startswith("0x") else hex(int(x))
        except ValueError:
            return x
    return x

def entropy_from_counts(counter):
    total = sum(counter.values())
    if total == 0: return 0.0
    probs = np.array(list(counter.values()), dtype=float) / total
    return float(-(probs * np.log(probs + 1e-9)).sum())

def read_block_structure(path):
    instr2block = {}
    block_heads_set = set()
    if not os.path.exists(path): return instr2block, []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pc, head = normalize_hex(parts[0]), normalize_hex(parts[1])
                instr2block[pc] = head
                block_heads_set.add(head)
    return instr2block, sorted(block_heads_set, key=lambda x: int(x, 16))

def read_ops(path):
    ops = defaultdict(list)
    if not os.path.exists(path): return ops
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ops[normalize_hex(parts[0])].append(parts[1].strip().upper())
    return ops

def read_entry_exit(entry_path, exit_path, instr2block):
    entries, exits = set(), set()
    for path, target_set in [(entry_path, entries), (exit_path, exits)]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    pc = normalize_hex(line.strip())
                    if pc and pc in instr2block: target_set.add(instr2block[pc])
    return entries, exits

def read_def_locations(path):
    defs = {}
    if not os.path.exists(path): return defs
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2: defs[parts[0]] = normalize_hex(parts[1])
    return defs



def convert_contract(fact_dir, label):
    instr2block, block_heads = read_block_structure(os.path.join(fact_dir, "block.facts"))
    if not block_heads: return None

    block_id  = {b: i for i, b in enumerate(block_heads)}
    num_nodes = len(block_heads)
    op_map   = read_ops(os.path.join(fact_dir, "op.facts"))
    entries, exits = read_entry_exit(
        os.path.join(fact_dir, "entry.facts"),
        os.path.join(fact_dir, "exit.facts"), instr2block
    )
    var_defs = read_def_locations(os.path.join(fact_dir, "def.facts"))

    block_instrs = defaultdict(list)
    for pc, blk in instr2block.items(): block_instrs[blk].append(pc)

    X = []
    VOCAB_SIZE = len(OPCODE_VOCAB) + 1

    for blk in block_heads:
        ops = []
        for pc in block_instrs[blk]: ops.extend(op_map.get(pc, []))
        counter = Counter(ops)

        feat = [0.0] * VOCAB_SIZE
        for op, c in counter.items():
            feat[opcode2idx.get(op, UNK_OPCODE_IDX)] += math.log1p(c)

        danger_count = 0
        for d in sorted(DANGEROUS_OPS):
            dc = counter[d]
            weight = CRITICAL_WEIGHTS.get(d, 1.0)
            feat.append(math.log1p(dc) * weight)
            danger_count += dc

        feat.append(math.log1p(len(ops)))
        feat.append(entropy_from_counts(counter))
        feat.append(danger_count / max(1, len(ops)))
        feat.append(1.0 if blk in entries else 0.0)
        feat.append(1.0 if blk in exits   else 0.0)

        X.append(feat)

    edges, edge_set = [], set()
    cfg_path = os.path.join(fact_dir, "CFGEdge.facts")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        ub, vb = instr2block.get(hex(int(parts[0]))), instr2block.get(hex(int(parts[1])))
                        if ub and vb and ub in block_id and vb in block_id:
                            s, t = block_id[ub], block_id[vb]
                            if (s, t) not in edge_set:
                                edges.append([s, t])
                                edge_set.add((s, t))
                    except (ValueError, KeyError): pass

    if not edges:
        edge_path = os.path.join(fact_dir, "edge.facts")
        if os.path.exists(edge_path):
            with open(edge_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            ub, vb = instr2block.get(normalize_hex(parts[0])), instr2block.get(normalize_hex(parts[1]))
                            if ub and vb and ub in block_id and vb in block_id:
                                s, t = block_id[ub], block_id[vb]
                                if (s, t) not in edge_set:
                                    edges.append([s, t])
                                    edge_set.add((s, t))
                        except (ValueError, KeyError): pass

    use_path = os.path.join(fact_dir, "use.facts")
    if os.path.exists(use_path):
        with open(use_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        def_pc = var_defs.get(parts[0])
                        if not def_pc: continue
                        ub, vb = instr2block.get(def_pc), instr2block.get(normalize_hex(parts[1]))
                        if ub and vb and ub != vb and ub in block_id and vb in block_id:
                            s, t = block_id[ub], block_id[vb]
                            if (s, t) not in edge_set:
                                edges.append([s, t])
                                edge_set.add((s, t))
                    except (KeyError, ValueError): pass

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), num_nodes=num_nodes)
    data.has_edges = edge_index.size(1) > 0
    data.num_edges = edge_index.size(1)
    return data

def normalize_features(graphs):
    all_x = torch.cat([g.x for g in graphs], dim=0).numpy()
    scaler = StandardScaler().fit(all_x)
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return graphs, scaler

def process_dataset():
    graphs, stats = [], defaultdict(int)
    stats['total_nodes'] = 0
    stats['total_edges'] = 0
    
    stats['benign_nodes'] = 0
    stats['benign_edges'] = 0
    stats['benign_count'] = 0
    stats['malicious_nodes'] = 0
    stats['malicious_edges'] = 0
    stats['malicious_count'] = 0
    
    for cls, label in [("benign", 0), ("malicious", 1)]:
        base = os.path.join(DATA_ROOT, cls)
        if not os.path.exists(base): continue
        for name in sorted(os.listdir(base)):
            contract_path = os.path.join(base, name)
            if not os.path.isdir(contract_path): continue
            stats["total"] += 1
            try:
                g = convert_contract(contract_path, label)
                if g is not None:
                    graphs.append(g)
                    stats["success"] += 1
                    stats["with_edges"] += int(g.has_edges)
                    stats['total_nodes'] += g.num_nodes
                    stats['total_edges'] += g.num_edges
                    
                    # 分类统计
                    if label == 0:
                        stats['benign_count'] += 1
                        stats['benign_nodes'] += g.num_nodes
                        stats['benign_edges'] += g.num_edges
                    else:
                        stats['malicious_count'] += 1
                        stats['malicious_nodes'] += g.num_nodes
                        stats['malicious_edges'] += g.num_edges
                else: 
                    stats["failed"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"Error processing {contract_path}: {e}")

    if not graphs: 
        print("No graphs were successfully processed!")
        return
    
    graphs, scaler = normalize_features(graphs)
    
    out_pt = os.path.join(OUTPUT_DIR, "graphs_fusion_final.pt")
    out_scaler = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
    torch.save(graphs, out_pt)
    import pickle
    with open(out_scaler, "wb") as f: 
        pickle.dump(scaler, f)
    
    
    avg_nodes = stats['total_nodes'] / stats['success'] if stats['success'] > 0 else 0
    avg_edges = stats['total_edges'] / stats['success'] if stats['success'] > 0 else 0
    
    
    avg_benign_nodes = stats['benign_nodes'] / stats['benign_count'] if stats['benign_count'] > 0 else 0
    avg_benign_edges = stats['benign_edges'] / stats['benign_count'] if stats['benign_count'] > 0 else 0
    avg_malicious_nodes = stats['malicious_nodes'] / stats['malicious_count'] if stats['malicious_count'] > 0 else 0
    avg_malicious_edges = stats['malicious_edges'] / stats['malicious_count'] if stats['malicious_count'] > 0 else 0
    
    print("\n=== Final statistics ===")
    print(f"Total: {stats['total']} | Success: {stats['success']} | Failed/Skipped: {stats['failed']}")
    print(f"Node feature dim: {graphs[0].num_node_features}")
    print(f"\n--- Overall ---")
    print(f"Total nodes: {stats['total_nodes']} | Total edges: {stats['total_edges']}")
    print(f"Average nodes per graph: {avg_nodes:.2f}")
    print(f"Average edges per graph: {avg_edges:.2f}")
    print(f"Graphs with edges: {stats['with_edges']}/{stats['success']} ({stats['with_edges']/stats['success']*100:.1f}%)")
    
    print(f"\n--- Benign Contracts ({stats['benign_count']}) ---")
    print(f"Average nodes: {avg_benign_nodes:.2f} | Average edges: {avg_benign_edges:.2f}")
    
    print(f"\n--- Malicious Contracts ({stats['malicious_count']}) ---")
    print(f"Average nodes: {avg_malicious_nodes:.2f} | Average edges: {avg_malicious_edges:.2f}")
    
    print(f"\nSaved graph data: {out_pt}")
    print(f"Saved scaler: {out_scaler}")

if __name__ == "__main__":
    process_dataset()