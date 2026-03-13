"""Diagnose texture rendering issues in a rigged GLB output.

Usage:
    python diagnose_texture.py <output.glb>

Checks every link in the chain:
  material → texture → image → bufferView → binary blob
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pygltflib


def check(cond, label, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return cond


def main(glb_path: str):
    path = Path(glb_path)
    print(f"\n=== Inspecting: {path} ===\n")

    gltf = pygltflib.GLTF2.load(str(path))
    blob = gltf.binary_blob()

    print(f"buffers      : {len(gltf.buffers)}")
    print(f"bufferViews  : {len(gltf.bufferViews)}")
    print(f"accessors    : {len(gltf.accessors)}")
    print(f"materials    : {len(gltf.materials)}")
    print(f"textures     : {len(gltf.textures)}")
    print(f"images       : {len(gltf.images)}")
    print(f"samplers     : {len(gltf.samplers)}")
    print(f"meshes       : {len(gltf.meshes)}")
    print(f"binary blob  : {len(blob) if blob else 0} bytes")
    print()

    # ── Per-mesh primitive check ──────────────────────────────────────────────
    for mi, mesh in enumerate(gltf.meshes):
        print(f"--- Mesh {mi}: '{mesh.name}' ---")
        for pi, prim in enumerate(mesh.primitives or []):
            mat_idx = prim.material
            has_pos = prim.attributes.POSITION is not None
            has_uv = prim.attributes.TEXCOORD_0 is not None
            has_norm = prim.attributes.NORMAL is not None
            print(f"  Primitive {pi}: material={mat_idx}  POSITION={has_pos}  "
                  f"TEXCOORD_0={has_uv}  NORMAL={has_norm}")

            # UV accessor validity
            if has_uv:
                acc_idx = prim.attributes.TEXCOORD_0
                acc = gltf.accessors[acc_idx]
                bv = gltf.bufferViews[acc.bufferView]
                byte_offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
                expected_bytes = acc.count * 2 * 4
                actual_bytes = bv.byteLength
                check(acc.type == "VEC2", f"    UV accessor type = VEC2", f"got {acc.type}")
                check(acc.count > 0, f"    UV count > 0", f"count={acc.count}")
                check(
                    blob is not None and byte_offset + expected_bytes <= len(blob),
                    f"    UV bytes in blob",
                    f"offset={byte_offset} need={expected_bytes} blob={len(blob) if blob else 0}",
                )
            else:
                print("    [FAIL] No TEXCOORD_0 — texture cannot be rendered on this primitive!")

            if mat_idx is None:
                print("    [FAIL] No material assigned — primitive will render gray!")
                continue

            # Material chain
            if mat_idx >= len(gltf.materials):
                print(f"    [FAIL] material index {mat_idx} out of range "
                      f"(only {len(gltf.materials)} materials)")
                continue

            mat = gltf.materials[mat_idx]
            print(f"    Material '{mat.name}': pbrMetallicRoughness="
                  f"{mat.pbrMetallicRoughness is not None}")

            pbr = mat.pbrMetallicRoughness
            if pbr is None:
                print("    [WARN] No pbrMetallicRoughness block")
                continue

            bc = pbr.baseColorTexture
            if bc is None:
                print("    [WARN] No baseColorTexture — material has no texture (solid color only)")
                if pbr.baseColorFactor:
                    print(f"    baseColorFactor = {pbr.baseColorFactor}")
                continue

            tex_idx = bc.index
            check(tex_idx is not None, "    baseColorTexture.index is set", f"index={tex_idx}")
            if tex_idx is None or tex_idx >= len(gltf.textures):
                print(f"    [FAIL] texture index {tex_idx} out of range "
                      f"(only {len(gltf.textures)} textures)")
                continue

            tex = gltf.textures[tex_idx]
            img_idx = tex.source
            check(img_idx is not None, f"    texture.source is set", f"img_idx={img_idx}")
            if img_idx is None or img_idx >= len(gltf.images):
                print(f"    [FAIL] image index {img_idx} out of range "
                      f"(only {len(gltf.images)} images)")
                continue

            img = gltf.images[img_idx]
            bv_idx = img.bufferView
            print(f"    Image {img_idx}: name='{img.name}'  mimeType={img.mimeType}  "
                  f"bufferView={bv_idx}  uri={img.uri}")

            if img.uri:
                print("    [INFO] Image is URI-referenced (not embedded) — skipping blob check")
                continue

            check(bv_idx is not None, "    image.bufferView is set")
            if bv_idx is None or bv_idx >= len(gltf.bufferViews):
                print(f"    [FAIL] bufferView index {bv_idx} out of range "
                      f"(only {len(gltf.bufferViews)} bufferViews)")
                continue

            bv = gltf.bufferViews[bv_idx]
            img_offset = bv.byteOffset or 0
            img_len = bv.byteLength
            print(f"    bufferView {bv_idx}: offset={img_offset}  length={img_len}")

            check(blob is not None, "    binary blob exists")
            if blob:
                in_range = img_offset + img_len <= len(blob)
                check(in_range, "    image bytes in blob",
                      f"offset={img_offset} len={img_len} blob={len(blob)}")

                if in_range and img_len > 0:
                    raw = blob[img_offset: img_offset + img_len]
                    # Check PNG or JPEG magic bytes
                    is_png = raw[:4] == b'\x89PNG'
                    is_jpeg = raw[:2] == b'\xff\xd8'
                    is_ktx = raw[:4] == b'\xabKTX'
                    check(is_png or is_jpeg or is_ktx, "    image magic bytes valid",
                          f"first4={raw[:4].hex()}")
                    print(f"    Image format: {'PNG' if is_png else 'JPEG' if is_jpeg else 'KTX2' if is_ktx else 'UNKNOWN'}")

        print()

    # ── Accessor range check ──────────────────────────────────────────────────
    print("--- Accessor blob-range check (first 5 and last 5) ---")
    n = len(gltf.accessors)
    indices = list(range(min(5, n))) + list(range(max(0, n - 5), n))
    seen = set()
    errors = 0
    for ai in indices:
        if ai in seen:
            continue
        seen.add(ai)
        acc = gltf.accessors[ai]
        if acc.bufferView is None:
            continue
        bv = gltf.bufferViews[acc.bufferView]
        offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        comp_bytes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}.get(acc.componentType, 4)
        type_comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}.get(acc.type, 1)
        needed = acc.count * comp_bytes * type_comps
        in_range = blob is not None and offset + needed <= len(blob)
        if not in_range:
            print(f"  [FAIL] accessor {ai}: offset={offset} need={needed} "
                  f"blob={len(blob) if blob else 0}")
            errors += 1
    if errors == 0:
        print("  [OK  ] All sampled accessors are in-range")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_texture.py <output.glb>")
        sys.exit(1)
    main(sys.argv[1])
