# -*- coding: utf-8 -*-
"""
Alchemy pending full transaction object version (with boundary protection and exception fallback)
pending pool -> creation input(initcode) -> local restore runtime -> Vandal -> GNN

Dependencies:
pip install web3 websockets eth-hash

Description:
1. Use eth_subscribe to subscribe to alchemy_pendingTransactions
2. Receive complete pending transaction object directly (no second get_transaction)
3. Filter contract creation transactions (to == None)
4. Read creation code from input
5. Execute creation code locally and attempt to recover RETURNed runtime bytecode
6. Save runtime .hex
7. Call vandal-master to generate CFG
8. Call GNN/detect.py for detection
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import importlib
from dataclasses import dataclass
from typing import Optional, Dict, List

import websockets
from web3 import Web3


# =========================
# Configuration
# =========================
FYP_ROOT = os.path.dirname(os.path.abspath(__file__))
VANDAL_ROOT = os.path.join(FYP_ROOT, "vandal-master")
HEX_INPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_input_hex")
CFG_OUTPUT_DIR = os.path.join(VANDAL_ROOT, "contracts_output_cfg")
GNN_DIR = os.path.join(FYP_ROOT, "GNN")
LOG_DIR = os.path.join(FYP_ROOT, "logs")

RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/iMNPA4ol4G9MzbMFo5mQ0"
WSS_URL = "wss://eth-mainnet.g.alchemy.com/v2/iMNPA4ol4G9MzbMFo5mQ0"

MAX_WORKERS = 4
STEP_LIMIT = 200000

# Added: boundary protection
MAX_MEMORY_BYTES = 16 * 1024 * 1024   # 16 MB
MAX_COPY_BYTES = 4 * 1024 * 1024      # 4 MB
MAX_RETURN_BYTES = 4 * 1024 * 1024    # 4 MB

# False: skip when extraction fails (recommended; better suited for reuse with existing runtime-GNN)
# True: fallback to direct initcode analysis when extraction fails
FALLBACK_TO_INITCODE = False

# Allow simulating external calls in local pre-execution; default off for safety
ALLOW_EXTERNAL_CALLS = False
# =========================


w3 = Web3(Web3.HTTPProvider(RPC_URL))

U256_MOD = 1 << 256
U256_MASK = U256_MOD - 1

try:
    from eth_hash.auto import keccak as keccak256
except Exception:
    keccak256 = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def append_jsonl(filename, record: dict):
    ensure_dir(LOG_DIR)
    path = os.path.join(LOG_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def u256(x: int) -> int:
    return x & U256_MASK


def to_signed(x: int) -> int:
    x &= U256_MASK
    return x if x < (1 << 255) else x - U256_MOD


def from_signed(x: int) -> int:
    return x & U256_MASK


def hex_to_bytes(hex_str: str) -> bytes:
    if not hex_str:
        return b""
    if isinstance(hex_str, bytes):
        return hex_str
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) % 2 == 1:
        hex_str = "0" + hex_str
    return bytes.fromhex(hex_str)


def bytes_to_hex(b: bytes) -> str:
    return "0x" + b.hex()


def to_hex_str(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        h = v.hex()
        return h if h.startswith("0x") else "0x" + h
    except Exception:
        s = str(v)
        return s if s.startswith("0x") else "0x" + s


def parse_quantity(v, default=0):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            if v.startswith("0x"):
                return int(v, 16)
            return int(v)
        except Exception:
            return default
    return default


def to_int_addr(addr) -> int:
    if not addr:
        return 0
    if isinstance(addr, int):
        return addr
    if isinstance(addr, str):
        try:
            return int(addr, 16)
        except Exception:
            return 0
    return 0


@dataclass
class ExtractionResult:
    success: bool
    runtime_hex: Optional[str]
    reason: str
    steps: int
    reverted: bool = False


class CreateRuntimeExtractor:
    def __init__(
        self,
        creation_input_hex: str,
        *,
        callvalue: int = 0,
        caller: int = 0,
        origin: int = 0,
        address: int = 0,
        gasprice: int = 0,
        chainid: int = 1,
        basefee: int = 0,
        coinbase: int = 0,
        timestamp: int = 0,
        number: int = 0,
        prevrandao: int = 0,
        gaslimit: int = 0,
        difficulty: int = 0,
        calldata: bytes = b"",
        step_limit: int = STEP_LIMIT,
        allow_external_calls: bool = False,
    ):
        self.code = hex_to_bytes(creation_input_hex)
        self.pc = 0
        self.stack: List[int] = []
        self.memory = bytearray()
        self.storage: Dict[int, int] = {}
        self.transient_storage: Dict[int, int] = {}
        self.steps = 0
        self.step_limit = step_limit
        self.halted = False
        self.reverted = False
        self.reason = ""
        self.return_data = b""
        self.last_return_data = b""
        self.allow_external_calls = allow_external_calls

        self.env = {
            "callvalue": u256(callvalue),
            "caller": u256(caller),
            "origin": u256(origin),
            "address": u256(address),
            "gasprice": u256(gasprice),
            "chainid": u256(chainid),
            "basefee": u256(basefee),
            "coinbase": u256(coinbase),
            "timestamp": u256(timestamp),
            "number": u256(number),
            "prevrandao": u256(prevrandao),
            "gaslimit": u256(gaslimit),
            "difficulty": u256(difficulty),
            "calldata": calldata or b"",
        }

    def fail(self, reason: str):
        self.halted = True
        self.reason = reason

    def ensure_mem(self, end_pos: int):
        if end_pos < 0:
            self.fail("negative memory access")
            return
        if end_pos > MAX_MEMORY_BYTES:
            self.fail(f"memory limit exceeded: {end_pos} > {MAX_MEMORY_BYTES}")
            return
        if end_pos > len(self.memory):
            self.memory.extend(b"\x00" * (end_pos - len(self.memory)))

    def pop(self) -> int:
        if not self.stack:
            self.fail("stack underflow")
            return 0
        return self.stack.pop()

    def push(self, x: int):
        self.stack.append(u256(x))

    def peek(self, n: int) -> int:
        if len(self.stack) < n:
            self.fail("stack underflow")
            return 0
        return self.stack[-n]

    def read_mem(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0:
            self.fail("negative memory read")
            return b""
        if size > MAX_COPY_BYTES:
            self.fail(f"memory read too large: {size}")
            return b""
        self.ensure_mem(offset + size)
        if self.halted:
            return b""
        return bytes(self.memory[offset:offset + size])

    def write_mem(self, offset: int, data: bytes):
        if offset < 0:
            self.fail("negative memory write")
            return
        if len(data) > MAX_COPY_BYTES:
            self.fail(f"memory write too large: {len(data)}")
            return
        self.ensure_mem(offset + len(data))
        if self.halted:
            return
        self.memory[offset:offset + len(data)] = data

    def mload(self, offset: int) -> int:
        return int.from_bytes(self.read_mem(offset, 32), "big")

    def mstore(self, offset: int, value: int):
        self.write_mem(offset, u256(value).to_bytes(32, "big"))

    def mstore8(self, offset: int, value: int):
        self.write_mem(offset, bytes([value & 0xFF]))

    def codecopy(self, mem_offset: int, code_offset: int, size: int):
        if mem_offset < 0 or code_offset < 0 or size < 0:
            self.fail("invalid CODECOPY args")
            return
        if size > MAX_COPY_BYTES:
            self.fail(f"CODECOPY size too large: {size}")
            return
        end_mem = mem_offset + size
        if end_mem > MAX_MEMORY_BYTES:
            self.fail(f"CODECOPY memory expansion too large: {end_mem}")
            return

        seg = self.code[code_offset:code_offset + size]
        if len(seg) < size:
            pad_len = size - len(seg)
            if pad_len > MAX_COPY_BYTES:
                self.fail(f"CODECOPY zero padding too large: {pad_len}")
                return
            seg += b"\x00" * pad_len

        self.write_mem(mem_offset, seg)

    def calldatacopy(self, mem_offset: int, data_offset: int, size: int):
        if mem_offset < 0 or data_offset < 0 or size < 0:
            self.fail("invalid CALLDATACOPY args")
            return
        if size > MAX_COPY_BYTES:
            self.fail(f"CALLDATACOPY size too large: {size}")
            return
        end_mem = mem_offset + size
        if end_mem > MAX_MEMORY_BYTES:
            self.fail(f"CALLDATACOPY memory expansion too large: {end_mem}")
            return

        data = self.env["calldata"][data_offset:data_offset + size]
        if len(data) < size:
            pad_len = size - len(data)
            if pad_len > MAX_COPY_BYTES:
                self.fail(f"CALLDATACOPY zero padding too large: {pad_len}")
                return
            data += b"\x00" * pad_len

        self.write_mem(mem_offset, data)

    def returndatacopy(self, mem_offset: int, data_offset: int, size: int):
        if mem_offset < 0 or data_offset < 0 or size < 0:
            self.fail("invalid RETURNDATACOPY args")
            return
        if size > MAX_COPY_BYTES:
            self.fail(f"RETURNDATACOPY size too large: {size}")
            return
        end_mem = mem_offset + size
        if end_mem > MAX_MEMORY_BYTES:
            self.fail(f"RETURNDATACOPY memory expansion too large: {end_mem}")
            return

        data = self.last_return_data[data_offset:data_offset + size]
        if len(data) < size:
            self.fail("RETURNDATACOPY out of bounds")
            return

        self.write_mem(mem_offset, data)

    def valid_jumpdest(self, dest: int) -> bool:
        return 0 <= dest < len(self.code) and self.code[dest] == 0x5B

    def op_binary(self, fn):
        a = self.pop()
        b = self.pop()
        if self.halted:
            return
        self.push(fn(b, a))

    def op_compare(self, fn):
        a = self.pop()
        b = self.pop()
        if self.halted:
            return
        self.push(1 if fn(b, a) else 0)

    def run(self) -> ExtractionResult:
        try:
            while not self.halted:
                if self.steps >= self.step_limit:
                    self.fail("step limit exceeded")
                    break

                if self.pc < 0 or self.pc >= len(self.code):
                    self.fail("pc out of range")
                    break

                self.steps += 1
                op = self.code[self.pc]
                self.pc += 1

                if op == 0x00:  # STOP
                    self.halted = True
                    self.reason = "STOP before RETURN"
                    break

                elif op == 0x5F:  # PUSH0
                    self.push(0)

                elif 0x60 <= op <= 0x7F:  # PUSH1..PUSH32
                    n = op - 0x5F
                    if self.pc + n > len(self.code):
                        self.fail("truncated PUSH data")
                        break
                    data = self.code[self.pc:self.pc + n]
                    self.pc += n
                    self.push(int.from_bytes(data, "big"))

                elif 0x80 <= op <= 0x8F:  # DUP1..DUP16
                    n = op - 0x7F
                    self.push(self.peek(n))

                elif 0x90 <= op <= 0x9F:  # SWAP1..SWAP16
                    n = op - 0x8F
                    if len(self.stack) < n + 1:
                        self.fail("stack underflow on SWAP")
                        break
                    self.stack[-1], self.stack[-1 - n] = self.stack[-1 - n], self.stack[-1]

                elif 0xA0 <= op <= 0xA4:  # LOG0..LOG4
                    topics = op - 0xA0
                    mem_offset = self.pop()
                    size = self.pop()
                    for _ in range(topics):
                        self.pop()
                    if self.halted:
                        break
                    self.read_mem(mem_offset, size)

                elif op == 0x01:  # ADD
                    self.op_binary(lambda x, y: x + y)
                elif op == 0x02:  # MUL
                    self.op_binary(lambda x, y: x * y)
                elif op == 0x03:  # SUB
                    self.op_binary(lambda x, y: x - y)
                elif op == 0x04:  # DIV
                    self.op_binary(lambda x, y: 0 if y == 0 else x // y)
                elif op == 0x05:  # SDIV
                    a = self.pop()
                    b = self.pop()
                    if self.halted:
                        break
                    sa, sb = to_signed(a), to_signed(b)
                    if sa == 0:
                        self.push(0)
                    elif sb == -(1 << 255) and sa == -1:
                        self.push(from_signed(-(1 << 255)))
                    else:
                        self.push(from_signed(int(sb / sa)))
                elif op == 0x06:  # MOD
                    self.op_binary(lambda x, y: 0 if y == 0 else x % y)
                elif op == 0x07:  # SMOD
                    a = self.pop()
                    b = self.pop()
                    if self.halted:
                        break
                    sa, sb = to_signed(a), to_signed(b)
                    if sa == 0:
                        self.push(0)
                    else:
                        sign = -1 if sb < 0 else 1
                        self.push(from_signed(sign * (abs(sb) % abs(sa))))
                elif op == 0x08:  # ADDMOD
                    a = self.pop()
                    b = self.pop()
                    n = self.pop()
                    if self.halted:
                        break
                    self.push(0 if n == 0 else (b + a) % n)
                elif op == 0x09:  # MULMOD
                    a = self.pop()
                    b = self.pop()
                    n = self.pop()
                    if self.halted:
                        break
                    self.push(0 if n == 0 else (b * a) % n)
                elif op == 0x0A:  # EXP
                    a = self.pop()
                    b = self.pop()
                    if self.halted:
                        break
                    self.push(pow(b, a, U256_MOD))
                elif op == 0x0B:  # SIGNEXTEND
                    b = self.pop()
                    x = self.pop()
                    if self.halted:
                        break
                    if b >= 32:
                        self.push(x)
                    else:
                        bit_index = 8 * b + 7
                        sign_bit = 1 << bit_index
                        mask = (1 << (bit_index + 1)) - 1
                        if x & sign_bit:
                            self.push(x | (U256_MASK ^ mask))
                        else:
                            self.push(x & mask)

                elif op == 0x10:  # LT
                    self.op_compare(lambda x, y: x < y)
                elif op == 0x11:  # GT
                    self.op_compare(lambda x, y: x > y)
                elif op == 0x12:  # SLT
                    self.op_compare(lambda x, y: to_signed(x) < to_signed(y))
                elif op == 0x13:  # SGT
                    self.op_compare(lambda x, y: to_signed(x) > to_signed(y))
                elif op == 0x14:  # EQ
                    self.op_compare(lambda x, y: x == y)
                elif op == 0x15:  # ISZERO
                    a = self.pop()
                    if self.halted:
                        break
                    self.push(1 if a == 0 else 0)
                elif op == 0x16:  # AND
                    self.op_binary(lambda x, y: x & y)
                elif op == 0x17:  # OR
                    self.op_binary(lambda x, y: x | y)
                elif op == 0x18:  # XOR
                    self.op_binary(lambda x, y: x ^ y)
                elif op == 0x19:  # NOT
                    a = self.pop()
                    if self.halted:
                        break
                    self.push(U256_MASK ^ a)
                elif op == 0x1A:  # BYTE
                    i = self.pop()
                    x = self.pop()
                    if self.halted:
                        break
                    if i >= 32:
                        self.push(0)
                    else:
                        shift = 8 * (31 - i)
                        self.push((x >> shift) & 0xFF)
                elif op == 0x1B:  # SHL
                    shift = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.push(0 if shift >= 256 else u256(value << shift))
                elif op == 0x1C:  # SHR
                    shift = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.push(0 if shift >= 256 else value >> shift)
                elif op == 0x1D:  # SAR
                    shift = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    sval = to_signed(value)
                    if shift >= 256:
                        self.push(U256_MASK if sval < 0 else 0)
                    else:
                        self.push(from_signed(sval >> shift))

                elif op == 0x20:  # KECCAK256
                    offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    if keccak256 is None:
                        self.fail("KECCAK256 unavailable (pip install eth-hash)")
                        break
                    data = self.read_mem(offset, size)
                    if self.halted:
                        break
                    self.push(int.from_bytes(keccak256(data), "big"))

                elif op == 0x30:  # ADDRESS
                    self.push(self.env["address"])
                elif op == 0x31:  # BALANCE
                    self.pop()
                    if self.halted:
                        break
                    self.push(0)
                elif op == 0x32:  # ORIGIN
                    self.push(self.env["origin"])
                elif op == 0x33:  # CALLER
                    self.push(self.env["caller"])
                elif op == 0x34:  # CALLVALUE
                    self.push(self.env["callvalue"])
                elif op == 0x35:  # CALLDATALOAD
                    offset = self.pop()
                    if self.halted:
                        break
                    data = self.env["calldata"][offset:offset + 32]
                    data += b"\x00" * (32 - len(data))
                    self.push(int.from_bytes(data, "big"))
                elif op == 0x36:  # CALLDATASIZE
                    self.push(len(self.env["calldata"]))
                elif op == 0x37:  # CALLDATACOPY
                    mem_offset = self.pop()
                    data_offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    self.calldatacopy(mem_offset, data_offset, size)
                elif op == 0x38:  # CODESIZE
                    self.push(len(self.code))
                elif op == 0x39:  # CODECOPY
                    mem_offset = self.pop()
                    code_offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    self.codecopy(mem_offset, code_offset, size)
                elif op == 0x3A:  # GASPRICE
                    self.push(self.env["gasprice"])
                elif op == 0x3B:  # EXTCODESIZE
                    self.pop()
                    if self.halted:
                        break
                    self.push(0)
                elif op == 0x3C:  # EXTCODECOPY
                    self.pop()
                    mem_offset = self.pop()
                    self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    if size > MAX_COPY_BYTES:
                        self.fail(f"EXTCODECOPY size too large: {size}")
                        break
                    self.write_mem(mem_offset, b"\x00" * size)
                elif op == 0x3D:  # RETURNDATASIZE
                    self.push(len(self.last_return_data))
                elif op == 0x3E:  # RETURNDATACOPY
                    mem_offset = self.pop()
                    data_offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    self.returndatacopy(mem_offset, data_offset, size)
                elif op == 0x3F:  # EXTCODEHASH
                    self.pop()
                    if self.halted:
                        break
                    self.push(0)

                elif op == 0x40:  # BLOCKHASH
                    self.pop()
                    if self.halted:
                        break
                    self.push(0)
                elif op == 0x41:  # COINBASE
                    self.push(self.env["coinbase"])
                elif op == 0x42:  # TIMESTAMP
                    self.push(self.env["timestamp"])
                elif op == 0x43:  # NUMBER
                    self.push(self.env["number"])
                elif op == 0x44:  # PREVRANDAO / DIFFICULTY
                    self.push(self.env["prevrandao"] or self.env["difficulty"])
                elif op == 0x45:  # GASLIMIT
                    self.push(self.env["gaslimit"])
                elif op == 0x46:  # CHAINID
                    self.push(self.env["chainid"])
                elif op == 0x47:  # SELFBALANCE
                    self.push(0)
                elif op == 0x48:  # BASEFEE
                    self.push(self.env["basefee"])

                elif op == 0x50:  # POP
                    self.pop()
                elif op == 0x51:  # MLOAD
                    offset = self.pop()
                    if self.halted:
                        break
                    self.push(self.mload(offset))
                elif op == 0x52:  # MSTORE
                    offset = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.mstore(offset, value)
                elif op == 0x53:  # MSTORE8
                    offset = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.mstore8(offset, value)
                elif op == 0x54:  # SLOAD
                    slot = self.pop()
                    if self.halted:
                        break
                    self.push(self.storage.get(slot, 0))
                elif op == 0x55:  # SSTORE
                    slot = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.storage[slot] = u256(value)
                elif op == 0x56:  # JUMP
                    dest = self.pop()
                    if self.halted:
                        break
                    if not self.valid_jumpdest(dest):
                        self.fail(f"invalid JUMP destination: {dest}")
                        break
                    self.pc = dest
                elif op == 0x57:  # JUMPI
                    dest = self.pop()
                    cond = self.pop()
                    if self.halted:
                        break
                    if cond != 0:
                        if not self.valid_jumpdest(dest):
                            self.fail(f"invalid JUMPI destination: {dest}")
                            break
                        self.pc = dest
                elif op == 0x58:  # PC
                    self.push(self.pc - 1)
                elif op == 0x59:  # MSIZE
                    self.push(len(self.memory))
                elif op == 0x5A:  # GAS
                    self.push((1 << 63) - 1)
                elif op == 0x5B:  # JUMPDEST
                    pass
                elif op == 0x5C:  # TLOAD
                    slot = self.pop()
                    if self.halted:
                        break
                    self.push(self.transient_storage.get(slot, 0))
                elif op == 0x5D:  # TSTORE
                    slot = self.pop()
                    value = self.pop()
                    if self.halted:
                        break
                    self.transient_storage[slot] = u256(value)
                elif op == 0x5E:  # MCOPY
                    dst = self.pop()
                    src = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    if size > MAX_COPY_BYTES:
                        self.fail(f"MCOPY size too large: {size}")
                        break
                    data = self.read_mem(src, size)
                    if self.halted:
                        break
                    self.write_mem(dst, data)

                elif op == 0xF0:  # CREATE
                    if not self.allow_external_calls:
                        self.fail("CREATE unsupported in safe mode")
                        break
                    self.pop()
                    self.pop()
                    self.pop()
                    if self.halted:
                        break
                    self.last_return_data = b""
                    self.push(0)

                elif op == 0xF1:  # CALL
                    if not self.allow_external_calls:
                        self.fail("CALL unsupported in safe mode")
                        break
                    self.pop()
                    self.pop()
                    self.pop()
                    self.pop()
                    self.pop()
                    out_offset = self.pop()
                    out_size = self.pop()
                    if self.halted:
                        break
                    if out_size > MAX_COPY_BYTES:
                        self.fail(f"CALL out_size too large: {out_size}")
                        break
                    self.last_return_data = b""
                    self.write_mem(out_offset, b"\x00" * out_size)
                    self.push(0)

                elif op == 0xF2:  # CALLCODE
                    self.fail("CALLCODE unsupported")
                    break

                elif op == 0xF3:  # RETURN
                    offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    if offset < 0 or size < 0:
                        self.fail("invalid RETURN args")
                        break
                    if size > MAX_RETURN_BYTES:
                        self.fail(f"RETURN size too large: {size}")
                        break
                    if offset + size > MAX_MEMORY_BYTES:
                        self.fail(f"RETURN memory range too large: {offset + size}")
                        break
                    self.return_data = self.read_mem(offset, size)
                    if self.halted:
                        break
                    self.halted = True
                    self.reason = "RETURN"
                    break

                elif op == 0xF4:  # DELEGATECALL
                    self.fail("DELEGATECALL unsupported")
                    break

                elif op == 0xF5:  # CREATE2
                    if not self.allow_external_calls:
                        self.fail("CREATE2 unsupported in safe mode")
                        break
                    self.pop()
                    self.pop()
                    self.pop()
                    self.pop()
                    if self.halted:
                        break
                    self.last_return_data = b""
                    self.push(0)

                elif op == 0xFA:  # STATICCALL
                    if not self.allow_external_calls:
                        self.fail("STATICCALL unsupported in safe mode")
                        break
                    self.pop()
                    self.pop()
                    self.pop()
                    self.pop()
                    out_offset = self.pop()
                    out_size = self.pop()
                    if self.halted:
                        break
                    if out_size > MAX_COPY_BYTES:
                        self.fail(f"STATICCALL out_size too large: {out_size}")
                        break
                    self.last_return_data = b""
                    self.write_mem(out_offset, b"\x00" * out_size)
                    self.push(0)

                elif op == 0xFD:  # REVERT
                    offset = self.pop()
                    size = self.pop()
                    if self.halted:
                        break
                    if size > MAX_RETURN_BYTES:
                        self.fail(f"REVERT size too large: {size}")
                        break
                    self.last_return_data = self.read_mem(offset, size)
                    self.reverted = True
                    self.halted = True
                    self.reason = "REVERT"
                    break

                elif op == 0xFE:  # INVALID
                    self.halted = True
                    self.reason = "INVALID"
                    break

                elif op == 0xFF:  # SELFDESTRUCT
                    self.halted = True
                    self.reason = "SELFDESTRUCT before RETURN"
                    break

                else:
                    self.fail(f"unsupported opcode 0x{op:02x}")
                    break

        except OverflowError as e:
            self.fail(f"overflow error: {e}")
        except MemoryError as e:
            self.fail(f"memory error: {e}")
        except Exception as e:
            self.fail(f"vm exception: {e}")

        if self.reason == "RETURN":
            return ExtractionResult(
                success=True,
                runtime_hex=bytes_to_hex(self.return_data),
                reason=self.reason,
                steps=self.steps,
                reverted=False,
            )

        return ExtractionResult(
            success=False,
            runtime_hex=None,
            reason=self.reason or "unknown failure",
            steps=self.steps,
            reverted=self.reverted,
        )


def extract_runtime_from_creation_input(
    creation_input_hex: str,
    *,
    callvalue: int = 0,
    caller: int = 0,
    origin: int = 0,
    address: int = 0,
    gasprice: int = 0,
    chainid: int = 1,
    basefee: int = 0,
    coinbase: int = 0,
    timestamp: int = 0,
    number: int = 0,
    prevrandao: int = 0,
    gaslimit: int = 0,
    difficulty: int = 0,
    calldata: bytes = b"",
    step_limit: int = STEP_LIMIT,
    allow_external_calls: bool = False,
) -> ExtractionResult:
    vm = CreateRuntimeExtractor(
        creation_input_hex,
        callvalue=callvalue,
        caller=caller,
        origin=origin,
        address=address,
        gasprice=gasprice,
        chainid=chainid,
        basefee=basefee,
        coinbase=coinbase,
        timestamp=timestamp,
        number=number,
        prevrandao=prevrandao,
        gaslimit=gaslimit,
        difficulty=difficulty,
        calldata=calldata,
        step_limit=step_limit,
        allow_external_calls=allow_external_calls,
    )
    return vm.run()


def save_hex(name, code_hex):
    ensure_dir(HEX_INPUT_DIR)
    path = os.path.join(HEX_INPUT_DIR, f"{name}.hex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code_hex)
    print(f"[✓] Saved {name}.hex -> {path}")
    return path


def clean_and_create_dir(dir_path):
    """Clean and recreate the directory to ensure fresh output for each decompile."""
    import shutil
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

async def subscribe_pending_transactions():
    seen_hashes = set()
    tx_count = 0  # counter

    async for ws in websockets.connect(WSS_URL):
        try:
            subscription_msg = {
                "jsonrpc": "2.0", "id": 1, "method": "eth_subscribe",
                "params": ["alchemy_pendingTransactions"]
            }
            await ws.send(json.dumps(subscription_msg))
            await ws.recv()
            print("[✓] Successfully subscribed to alchemy_pendingTransactions")

            while True:
                raw_msg = await ws.recv()
                msg = json.loads(raw_msg)
                tx = msg.get("params", {}).get("result")
                
                if not isinstance(tx, dict): continue
                
                tx_count += 1
                # Print a dot every 100 regular transactions to show the program is alive
                if tx_count % 100 == 0:
                    print(".", end="", flush=True)

                if tx.get("to") is not None: continue

                tx_hash = tx.get("hash")
                if not tx_hash or tx_hash in seen_hashes: continue

                seen_hashes.add(tx_hash)
                # trigger detection
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, process_pending_tx_object, tx)

        except Exception as e:
            print(f"\n[!] WebSocket error: {e}")
            await asyncio.sleep(3)

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


def run_gnn_detect(facts_dir, return_result=False):
    if not os.path.isdir(GNN_DIR):
        print(f"[!] GNN directory not found: {GNN_DIR}")
        return None

    if GNN_DIR not in sys.path:
        sys.path.insert(0, GNN_DIR)

    orig_cwd = os.getcwd()
    try:
        os.chdir(GNN_DIR)
        import detect
        importlib.reload(detect)

        if return_result:
            return detect.run_detection_result(target_facts_dir=facts_dir)
        detect.run_detection(target_facts_dir=facts_dir)
        return None
    except Exception as e:
        print(f"[!] GNN detection error: {e}")
        return None
    finally:
        os.chdir(orig_cwd)


def run_pipeline_from_runtime_hex(sample_name, runtime_hex):
    hex_path = save_hex(sample_name, runtime_hex)

    facts_dir = run_vandal_decompile(hex_path, sample_name)
    if not facts_dir or not os.path.isdir(facts_dir):
        return False, "CFG generation failed (Vandal decompilation unsuccessful)"

    result = run_gnn_detect(facts_dir, return_result=True)
    if result is None:
        return False, "GNN detection failed"

    return True, result


def process_pending_tx_object(tx: dict):
    try:
        if not isinstance(tx, dict):
            return

        if tx.get("to") is not None:
            return

        tx_hash = tx.get("hash")
        initcode_hex = to_hex_str(tx.get("input"))
        if not tx_hash or not initcode_hex or initcode_hex == "0x":
            return

        from_addr = tx.get("from")
        value = parse_quantity(tx.get("value"), 0)
        gas = parse_quantity(tx.get("gas"), 0)
        gasprice = parse_quantity(tx.get("gasPrice"), 0)

        print("\n" + "=" * 80)
        print("[Pending contract creation transaction found]")
        print(f"tx_hash : {tx_hash}")
        print(f"from    : {from_addr}")
        print(f"gas     : {gas}")
        print(f"value   : {value}")
        print("=" * 80)

        extract_result = extract_runtime_from_creation_input(
            initcode_hex,
            callvalue=value,
            caller=to_int_addr(from_addr),
            origin=to_int_addr(from_addr),
            address=0,
            gasprice=gasprice,
            chainid=1,
            calldata=b"",
            step_limit=STEP_LIMIT,
            allow_external_calls=ALLOW_EXTERNAL_CALLS,
        )

        if extract_result.success and extract_result.runtime_hex:
            runtime_len = (len(extract_result.runtime_hex) - 2) // 2
            sample_name = f"pending_runtime_{tx_hash[2:]}"
            print(f"[✓] Runtime extraction succeeded, length: {runtime_len} bytes, steps={extract_result.steps}")

            success, data = run_pipeline_from_runtime_hex(sample_name, extract_result.runtime_hex)
            if success:
                print(f"[✓] Detection completed: {tx_hash}")
                print(f"[Result] {data}")

                
                if data.get("score", 0) > 0.8:
                    
                    high_risk_log = "high_risk_alerts.jsonl" 
                    print(f"⚠️  [High-risk alert] Malicious score {data['score_pct']} exceeded threshold, logging...")
                    
                    append_jsonl(high_risk_log, {
                        "time": now_ts(),
                        "tx_hash": tx_hash,
                        "from": from_addr,
                        "score": data["score"],
                        "score_pct": data["score_pct"],
                        "conclusion": data["conclusion"]
                    })

            return

        print(f"[!] Runtime extraction failed: {extract_result.reason}, steps={extract_result.steps}")

        append_jsonl("runtime_extract_failures.jsonl", {
            "time": now_ts(),
            "tx_hash": tx_hash,
            "from": from_addr,
            "gas": gas,
            "value": value,
            "extract_success": False,
            "extract_reason": extract_result.reason,
            "extract_steps": extract_result.steps,
            "reverted": extract_result.reverted,
            "initcode_len": (len(initcode_hex) - 2) // 2,
        })

        if FALLBACK_TO_INITCODE:
            sample_name = f"pending_init_{tx_hash[2:]}"
            print("[*] Falling back to direct initcode analysis")
            success, data = run_pipeline_from_runtime_hex(sample_name, initcode_hex)
            if success:
                print(f"[✓] Fallback detection completed: {tx_hash}")
                print(f"[Result] {data}")
            else:
                print(f"[✗] Fallback detection failed: {tx_hash} -> {data}")
        else:
            print("[*] Transaction skipped")

    except Exception as e:
        tx_hash = tx.get("hash") if isinstance(tx, dict) else None
        print(f"[!] process_pending_tx_object error: {e}")
        append_jsonl("pending_errors.jsonl", {
            "time": now_ts(),
            "stage": "process_pending_tx_object",
            "tx_hash": tx_hash,
            "error": str(e),
        })

def main():
    print("=" * 88)
    print("  Smart contract pending detection pipeline (Alchemy full tx -> runtime -> CFG -> GNN)")
    print("=" * 88)

    if not w3.is_connected():
        print("[!] HTTP RPC connection failed. Check network or API key")
        return

    print("[*] HTTP RPC connection successful")

    # Important: ensure this function name matches the subscription function defined above
    try:
        asyncio.run(subscribe_pending_transactions()) 
    except KeyboardInterrupt:
        print("\n[*] User stopped execution.")


if __name__ == "__main__":
    main()
