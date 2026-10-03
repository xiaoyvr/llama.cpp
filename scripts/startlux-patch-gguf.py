#!/usr/bin/env python3
"""Add/override GGUF metadata keys without touching tensor data.

usage: patch_gguf.py <src.gguf> <dst.gguf> <noul_temp> <choice_temp> <score_temp>
"""
import struct, sys

MAGIC = b"GGUF"; ALIGN = 32
FMT = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),7:("<?",1),10:("<Q",8),11:("<q",8),12:("<d",8)}
STR, ARR, F32 = 8, 9, 6

# StartLux prompt, reproduces startlux_decision/jevfmt.py for the standard (non-bare) option ids
STARTLUX_TMPL = r"""{% set letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' %}<|im_start|>system
Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.<|im_end|>
<|im_start|>user
Evidence:
{{ state if state is string else state | tojson }}

Question: {{ instructions if instructions is string else instructions | tojson }}
Options:
{% for o in options %}{{ letters[loop.index0] }}) {% if type == 'noul' %}{{ 'yes' if o.key == 'true' else 'no' }}{% else %}{{ o.key }}{% endif %}{% if o.description %}: {{ o.description if o.description is string else o.description | tojson }}{% endif %}
{% endfor %}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)

def rd_val(f, t):
    if t == STR: return rd_str(f)
    if t == ARR:
        et = struct.unpack("<I", f.read(4))[0]; c = struct.unpack("<Q", f.read(8))[0]
        return (et, [rd_val(f, et) for _ in range(c)])
    fmt, sz = FMT[t]; return struct.unpack(fmt, f.read(sz))[0]

def wr_str(o, b):
    o.write(struct.pack("<Q", len(b))); o.write(b)

def wr_val(o, t, v):
    if t == STR: wr_str(o, v)
    elif t == ARR:
        et, items = v
        o.write(struct.pack("<I", et)); o.write(struct.pack("<Q", len(items)))
        for it in items: wr_val(o, et, it)
    else:
        fmt, sz = FMT[t]; o.write(struct.pack(fmt, v))

def main(src, dst, tn, tc, ts):
    with open(src, "rb") as f:
        assert f.read(4) == MAGIC
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        kvs = []
        arch = "qwen35"
        for _ in range(n_kv):
            k = rd_str(f); t = struct.unpack("<I", f.read(4))[0]; v = rd_val(f, t)
            kvs.append((k, t, v))
            if k == b"general.architecture": arch = v.decode()
        ti = bytearray()
        for _ in range(n_tensors):
            k = rd_str(f)
            ti += struct.pack("<Q", len(k)) + k
            nd = struct.unpack("<I", f.read(4))[0]; ti += struct.pack("<I", nd) + f.read(8*nd)
            ti += f.read(4) + f.read(8)
        data_pos = f.tell()
        pad = (ALIGN - (data_pos % ALIGN)) % ALIGN

    add = {
        b"tokenizer.chat_template.systemone": (STR, STARTLUX_TMPL.encode()),
        (arch + ".decision.type").encode(): (STR, b"startlux"),
        (arch + ".decision.temperature.noul").encode():   (F32, float(tn)),
        (arch + ".decision.temperature.choice").encode(): (F32, float(tc)),
        (arch + ".decision.temperature.score").encode():  (F32, float(ts)),
    }
    new, seen = [], set()
    for k, t, v in kvs:
        if k in add:
            nt, nv = add.pop(k); new.append((k, nt, nv)); seen.add(k)
        else:
            new.append((k, t, v))
    for k, (t, v) in add.items():
        new.append((k, t, v))

    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fi.seek(data_pos + pad)
        fo.write(MAGIC); fo.write(struct.pack("<I", version))
        fo.write(struct.pack("<Q", n_tensors)); fo.write(struct.pack("<Q", len(new)))
        for k, t, v in new:
            wr_str(fo, k); fo.write(struct.pack("<I", t)); wr_val(fo, t, v)
        fo.write(ti)
        fo.write(b"\x00" * ((ALIGN - (fo.tell() % ALIGN)) % ALIGN))
        while True:
            b = fi.read(8 << 20)
            if not b: break
            fo.write(b)
    print("patched ->", dst, "| arch =", arch, "| keys:", len(new))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
