# utils_release.py
import os
import requests

GITHUB_API = "https://api.github.com"

def _gh_headers():
    token = os.getenv("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def _dl_headers():
    token = os.getenv("GITHUB_TOKEN", "")
    h = {"Accept": "application/octet-stream"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def download_asset_from_latest(owner: str, repo: str, asset_name: str, out_path: str):
    """
    Descarga un asset por nombre desde el último release.
    Requiere GITHUB_TOKEN (lo inyecta GitHub Actions por defecto).
    """
    # 1) Pedir último release
    url = f"{GITHUB_API}/repos/{owner}/{repo}/releases/latest"
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    data = r.json()
    assets = data.get("assets", [])
    # 2) Buscar asset
    asset = next((a for a in assets if a.get("name") == asset_name), None)
    if not asset:
        raise RuntimeError(f"No se encontró el asset '{asset_name}' en el último release.")
    asset_url = asset.get("url")
    # 3) Descargar binario
    dr = requests.get(asset_url, headers=_dl_headers(), timeout=300)
    dr.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(dr.content)
    return out_path
