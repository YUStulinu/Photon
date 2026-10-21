#!/usr/bin/env python3
"""Downloads test assets into scenes/models and scenes/hdri (they are not part of the repository).

  - DamagedHelmet and Sponza from the Khronos glTF sample assets (CC BY 4.0 / CC BY 3.0),
  - FlightHelmet (several meshes, many textures) and the IOR/transmission test models,
  - two equirectangular HDR environments from Poly Haven (CC0).

usage: python scripts/fetch_models.py [name...]     (default: all)
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
SAMPLES_API = "https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/contents/Models"

# name -> (kind, source)
ASSETS = {
    "DamagedHelmet": ("file", "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"),
    "Sponza": ("dir", "Sponza/glTF"),
    "FlightHelmet": ("dir", "FlightHelmet/glTF"),
    "DragonAttenuation": ("file", "DragonAttenuation/glTF-Binary/DragonAttenuation.glb"),
    "kloppenheim": ("hdri", "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/kloppenheim_06_puresky_2k.hdr"),
    "studio": ("hdri", "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_09_2k.hdr"),
}


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "photon-fetch"})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(path + ".part", path)


def fetch_dir(api_path, out_dir):
    req = urllib.request.Request(f"{SAMPLES_API}/{api_path}", headers={"User-Agent": "photon-fetch"})
    with urllib.request.urlopen(req) as r:
        entries = json.load(r)
    for e in entries:
        if e["type"] == "file":
            download(e["download_url"], os.path.join(out_dir, e["name"]))
        elif e["type"] == "dir":
            fetch_dir(f"{api_path}/{e['name']}", os.path.join(out_dir, e["name"]))


def main():
    names = sys.argv[1:] or list(ASSETS)
    for name in names:
        if name not in ASSETS:
            sys.exit(f"unknown asset {name}; known: {', '.join(ASSETS)}")
        kind, src = ASSETS[name]
        print(name)
        if kind == "file":
            download(f"{SAMPLES}/{src}", os.path.join(ROOT, "scenes", "models", name, os.path.basename(src)))
        elif kind == "dir":
            fetch_dir(src, os.path.join(ROOT, "scenes", "models", name))
        else:
            download(src, os.path.join(ROOT, "scenes", "hdri", os.path.basename(src)))
    print("done")


if __name__ == "__main__":
    main()
