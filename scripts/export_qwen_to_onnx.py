#!/usr/bin/env python3
"""
Export the fine-tuned Qwen2.5-0.5B joke model to ONNX for the browser.

- Exports with a KV cache (Optimum task `text-generation-with-past`) so each
  generated token costs one small forward pass instead of re-processing the
  whole sequence. This is what brings per-joke latency down from ~11 s.
- Quantizes MatMul weights to 4-bit (MatMulNBits, block size 32), which
  ONNX Runtime Web >= 1.17 runs on both the wasm and webgpu backends.
- Writes a single self-contained .onnx (weights embedded) plus the tokenizer
  files the browser needs.

Usage:
    python scripts/finetune_qwen.py          # first
    python scripts/export_qwen_to_onnx.py [merged_model_dir] [output_dir]

    defaults: dad-joke-model/qwen_finetuned_hf -> dad-joke-model/qwen

Output:
    <output_dir>/model_q4.onnx               (+ tokenizer.json etc.)
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from optimum.exporters.onnx import main_export

MERGED_DIR = sys.argv[1] if len(sys.argv) > 1 else "dad-joke-model/qwen_finetuned_hf"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "dad-joke-model/qwen"
OUT_MODEL = os.path.join(OUT_DIR, "model_q4.onnx")
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "config.json", "generation_config.json"]


def untie_lm_head(model):
    """
    Optimum ties Qwen's output head to the embedding table as
    Transpose(embed_tokens.weight) -> MatMul. MatMulNBits only quantizes a
    MatMul whose weight is a direct initializer, so the head would stay fp32
    (and quantize_embedding_int8 would then remove the tensor it needs).
    Materialize the transposed weight as its own initializer and drop the
    Transpose, so the head is quantized to 4-bit like every other MatMul.
    """
    from onnx import numpy_helper

    inits = {t.name: t for t in model.graph.initializer}
    transposes = {n.output[0]: n for n in model.graph.node
                  if n.op_type == "Transpose" and n.input[0] in inits}
    done = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.input[1] not in transposes:
            continue
        tnode = transposes[node.input[1]]
        w = numpy_helper.to_array(inits[tnode.input[0]])
        perm = next((a.ints for a in tnode.attribute if a.name == "perm"), None)
        wt = np.ascontiguousarray(np.transpose(w, perm) if perm else w.T)
        new_name = tnode.input[0] + "_T"
        model.graph.initializer.append(numpy_helper.from_array(wt, new_name))
        node.input[1] = new_name
        model.graph.node.remove(tnode)
        done += 1
        print(f"  untied head: {node.name} now uses {new_name} {wt.shape}")
    return done


def quantize_embedding_int8(model, min_rows=50000):
    """
    MatMulNBits leaves the token-embedding Gather as fp32, and for Qwen's
    152k vocab that single tensor is ~545 MB. Replace it with a per-row
    symmetric int8 table:  Gather(int8) -> Cast(float) -> Mul(row scale).
    Returns the number of tables converted.
    """
    from onnx import helper, numpy_helper, TensorProto

    inits = {t.name: t for t in model.graph.initializer}
    converted = 0
    for node in list(model.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits:
            continue
        w = numpy_helper.to_array(inits[node.input[0]])
        if w.ndim != 2 or w.shape[0] < min_rows:
            continue

        scale = (np.abs(w).max(axis=1, keepdims=True) / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0
        wq = np.clip(np.round(w / scale), -127, 127).astype(np.int8)

        base = node.input[0]
        q_name, s_name = base + "_int8", base + "_row_scale"
        model.graph.initializer.remove(inits[base])
        model.graph.initializer.extend([
            numpy_helper.from_array(wq, q_name),
            numpy_helper.from_array(scale, s_name),
        ])

        out = node.output[0]
        node.input[0] = q_name
        node.output[0] = out + "_int8"
        new_nodes = [
            helper.make_node("Cast", [out + "_int8"], [out + "_f32"], to=TensorProto.FLOAT),
            helper.make_node("Gather", [s_name, node.input[1]], [out + "_scale"], axis=0),
            helper.make_node("Mul", [out + "_f32", out + "_scale"], [out]),
        ]
        idx = list(model.graph.node).index(node)
        for k, n in enumerate(new_nodes):
            model.graph.node.insert(idx + 1 + k, n)
        converted += 1
        print(f"  embedding {base}: {w.shape} fp32 -> int8 "
              f"({w.nbytes / 2**20:.0f} MB -> {wq.nbytes / 2**20:.0f} MB)")
    return converted


def fix_merges_for_transformers_js(tokenizer_json):
    """
    tokenizers >= 0.20 writes BPE merges as [["a","b"], ...]; Transformers.js
    2.x only understands the older "a b" string form and throws
    `e.split is not a function` otherwise. Rewrite in place.
    """
    import json
    with open(tokenizer_json) as f:
        t = json.load(f)
    merges = t.get("model", {}).get("merges", [])
    if merges and isinstance(merges[0], list):
        t["model"]["merges"] = [" ".join(pair) for pair in merges]
        with open(tokenizer_json, "w") as f:
            json.dump(t, f, ensure_ascii=False)
        print(f"  tokenizer.json: converted {len(merges)} merges to string form")


def describe(path):
    m = onnx.load(path, load_external_data=False)
    ins = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]
    print(f"  inputs ({len(ins)}): {ins[:4]} ...")
    print(f"  outputs ({len(outs)}): {outs[:3]} ...")
    return ins, outs


def main():
    if not os.path.isdir(MERGED_DIR):
        print(f"Merged model not found: {MERGED_DIR}\nRun scripts/finetune_qwen.py first.")
        sys.exit(1)

    print("=" * 70)
    print("ONNX export - fine-tuned Qwen2.5-0.5B (KV cache + 4-bit)")
    print("=" * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"\nExporting {MERGED_DIR} with Optimum (text-generation-with-past)...", flush=True)
        main_export(MERGED_DIR, output=tmp, task="text-generation-with-past")
        fp32 = os.path.join(tmp, "model.onnx")
        fp32_mb = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp) if f.startswith("model.onnx")) / 2**20
        print(f"  fp32 size: {fp32_mb:.0f} MB")
        ins, outs = describe(fp32)

        print("\nQuantizing MatMul weights to 4-bit (block size 32)...", flush=True)
        model = onnx.load(fp32)
        untie_lm_head(model)
        quant = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True, accuracy_level=4)
        quant.process()
        qmodel = quant.model.model

        print("Quantizing token embedding table to int8...", flush=True)
        if quantize_embedding_int8(qmodel) == 0:
            print("  (no large embedding Gather found; skipped)")
        onnx.save(qmodel, OUT_MODEL)

        for f in TOKENIZER_FILES:
            src = os.path.join(tmp, f)
            if os.path.exists(src):
                shutil.copy(src, OUT_DIR)
        fix_merges_for_transformers_js(os.path.join(OUT_DIR, "tokenizer.json"))

    q_mb = os.path.getsize(OUT_MODEL) / 2**20
    print(f"  4-bit size: {q_mb:.0f} MB -> {OUT_MODEL}")

    print("\nSmoke-testing quantized model (prefill + one cached step)...", flush=True)
    sess = ort.InferenceSession(OUT_MODEL)
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    n_layers = sum(1 for n in in_names if n.endswith(".key"))
    kv_shape = next(i.shape for i in sess.get_inputs() if i.name.endswith(".key"))
    n_kv_heads, head_dim = kv_shape[1], kv_shape[3]
    print(f"  layers={n_layers} kv_heads={n_kv_heads} head_dim={head_dim}")

    def empty_past(batch=1):
        return {n: np.zeros((batch, n_kv_heads, 0, head_dim), dtype=np.float32)
                for n in in_names if n.startswith("past_key_values")}

    prompt = np.array([[48, 25]], dtype=np.int64)          # "Q:"
    feeds = {"input_ids": prompt,
             "attention_mask": np.ones_like(prompt),
             "position_ids": np.arange(prompt.shape[1], dtype=np.int64)[None, :],
             **empty_past()}
    outs = sess.run(None, feeds)
    logits = outs[out_names.index("logits")]
    print(f"  prefill logits: {logits.shape}")

    # one decode step reusing the cache
    present = {n.replace("present", "past_key_values"): v for n, v in zip(out_names, outs) if n.startswith("present")}
    nxt = np.array([[int(logits[0, -1].argmax())]], dtype=np.int64)
    feeds = {"input_ids": nxt,
             "attention_mask": np.ones((1, prompt.shape[1] + 1), dtype=np.int64),
             "position_ids": np.array([[prompt.shape[1]]], dtype=np.int64),
             **present}
    outs = sess.run(None, feeds)
    print(f"  decode-step logits: {outs[out_names.index('logits')].shape}")
    print("  Inference OK!")

    print("\n" + "=" * 70)
    print(f"Export complete: {OUT_MODEL} ({q_mb:.0f} MB)")
    print("Next: update MODEL_PATH / tokenizer in index.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
