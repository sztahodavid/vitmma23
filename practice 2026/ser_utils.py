# -*- coding: utf-8 -*-
"""
ser_utils.py — segédfüggvények az érzelemfelismerés gyakorlathoz.

MI alapú ember-gép interakció · BME VIK
Emberi emóciók felismerése — 90 perces gyakorlat

Ezt a modult NEM az órán írjuk meg: előre kiadott segédkód, hogy a gyakorlaton
a lényegre — az adatra, a reprezentációra és a kiértékelésre — jusson idő.

Használat Colabban: a notebookok betöltő cellája gondoskodik róla. Kézzel:
    import sys, importlib
    sys.modules.pop("ser_utils", None)      # esetleges üres/régi modul eldobása
    import ser_utils as su
    su.self_check()
"""
from __future__ import annotations

__version__ = "1.8"

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Alapadatok
# --------------------------------------------------------------------------
RAVDESS_URL = ("https://zenodo.org/records/1188976/files/"
               "Audio_Speech_Actors_01-24.zip?download=1")
RAVDESS_ZIP = "Audio_Speech_Actors_01-24.zip"

# A RAVDESS fájlnév hét számból áll, kötőjellel elválasztva:
#   03-01-05-02-01-01-12.wav
#   |  |  |  |  |  |  └─ színész (01-24; páratlan = férfi, páros = nő)
#   |  |  |  |  |  └──── ismétlés (01, 02)
#   |  |  |  |  └─────── mondat (01: "Kids are talking by the door",
#   |  |  |  |                   02: "Dogs are sitting by the door")
#   |  |  |  └────────── intenzitás (01: normál, 02: erős)
#   |  |  └───────────── érzelem (lásd EMOTIONS)
#   |  └──────────────── vokális csatorna (01: beszéd, 02: ének)
#   └─────────────────── modalitás (01: audio+video, 02: video, 03: csak audio)
EMOTIONS = {
    "01": "semleges",
    "02": "nyugodt",
    "03": "öröm",
    "04": "szomorúság",
    "05": "harag",
    "06": "félelem",
    "07": "undor",
    "08": "meglepettség",
}
EMOTION_ORDER = ["semleges", "nyugodt", "öröm", "szomorúság",
                 "harag", "félelem", "undor", "meglepettség"]

SR = 16_000          # a beszédmodellek mintavételi frekvenciája
RAVDESS_SR = 48_000  # a RAVDESS eredeti mintavételi frekvenciája

_FNAME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")


# --------------------------------------------------------------------------
# 0. Környezet
# --------------------------------------------------------------------------
def in_colab() -> bool:
    return "google.colab" in sys.modules


def gpu_info() -> str:
    """Rövid jelentés arról, van-e használható GPU, és mennyi a memóriája."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return (f"GPU: {torch.cuda.get_device_name(0)} · {vram:.1f} GB VRAM "
                    f"· CUDA {torch.version.cuda}")
        return ("GPU: nincs (CPU-n fut) — az embedding-számítás 1440 felvételre "
                "kb. 15-20 perc; kisebb adathalmazzal érdemes dolgozni")
    except ImportError:
        return "GPU: a torch nincs telepítve"


def suggested_batch_size(default: int = 16) -> int:
    """A VRAM alapján javasolt batch méret az embedding-számításhoz."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 4
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram < 5:      # pl. 4 GB-os laptop GPU
            return 4
        if vram < 9:
            return 8
        return default
    except ImportError:
        return 4


def ensure_packages(quiet: bool = True) -> None:
    """A gyakorlathoz szükséges csomagok telepítése, ha hiányoznak."""
    need = []
    for mod, pkg in [("soundfile", "soundfile"), ("sklearn", "scikit-learn"),
                     ("torch", "torch"), ("transformers", "transformers"),
                     ("opensmile", "opensmile"), ("pandas", "pandas"),
                     ("matplotlib", "matplotlib"), ("datasets", "datasets"),
                     ("skimage", "scikit-image")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print("Telepítés:", ", ".join(need))
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("-q")
        subprocess.run(cmd + need, check=True)
    else:
        print("Minden szükséges csomag megvan.")


# --------------------------------------------------------------------------
# 1. Adat letöltése és beolvasása
# --------------------------------------------------------------------------
def _download(url: str, dest: str | Path) -> None:
    """Letöltés urllib-bel — wget nélkül is működik (Windows, macOS, Linux)."""
    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (ser_utils oktatasi celra)")]
    urllib.request.install_opener(opener)

    def hook(blocks, block_size, total):
        if total > 0:
            done = blocks * block_size
            pct = min(100, int(done * 100 / total))
            print(f"\r  {pct:3d}%  ({done / 1e6:6.0f} / {total / 1e6:.0f} MB)",
                  end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    print()


def download_ravdess(data_root: str | Path = "data", force: bool = False) -> Path:
    """Letölti és kicsomagolja a RAVDESS beszédkorpuszt (~208 MB, 1440 felvétel).

    Ha a cél könyvtárban már megvannak a fájlok, nem tölt le újra — ezért
    érdemes a Drive-ra mutató elérési utat megadni, és órán már csak beolvasni.
    """
    root = Path(data_root)
    wav_root = root / "ravdess"
    root.mkdir(parents=True, exist_ok=True)

    if wav_root.exists() and not force:
        n = len(list(wav_root.rglob("*.wav")))
        if n >= 1400:
            print(f"Az adat már megvan: {n} felvétel a(z) {wav_root} könyvtárban.")
            return wav_root

    zip_path = root / RAVDESS_ZIP
    if zip_path.exists() and zip_path.stat().st_size < 150_000_000 and not force:
        print("A korábbi letöltés csonka, újrakezdem.")
        zip_path.unlink()
    if not zip_path.exists() or force:
        print(f"Letöltés ({RAVDESS_ZIP}, ~208 MB)…")
        _download(RAVDESS_URL, zip_path)
        if zip_path.stat().st_size < 150_000_000:
            zip_path.unlink()
            raise RuntimeError(
                "A letöltés csonka lett. Próbáld újra, vagy töltsd le kézzel a "
                f"{RAVDESS_URL} címről, és tedd ide: {root}")
    print("Kicsomagolás…")
    wav_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(wav_root)
    n = len(list(wav_root.rglob("*.wav")))
    print(f"Kész: {n} felvétel a(z) {wav_root} könyvtárban.")
    return wav_root


def parse_filename(path: str | Path) -> dict | None:
    """A RAVDESS fájlnevéből kiszedi a metaadatokat. Ismeretlen névre None."""
    stem = Path(path).stem
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    modality, vocal, emo, intens, stmt, rep, actor = m.groups()
    actor_i = int(actor)
    return {
        "path": str(path),
        "modality": modality,
        "vocal_channel": vocal,
        "emotion_code": emo,
        "emotion": EMOTIONS.get(emo, "ismeretlen"),
        "intensity": "erős" if intens == "02" else "normál",
        "statement": int(stmt),
        "repetition": int(rep),
        "actor": actor_i,
        "gender": "férfi" if actor_i % 2 == 1 else "nő",
    }


def scan_ravdess(wav_root: str | Path):
    """Végigmegy a wav fájlokon, és egy pandas DataFrame-et ad vissza."""
    import pandas as pd
    rows = []
    for p in sorted(Path(wav_root).rglob("*.wav")):
        meta = parse_filename(p)
        if meta is not None:
            rows.append(meta)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Nem találtam RAVDESS wav fájlt itt: {wav_root}")
    return df.sort_values(["actor", "emotion_code"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Hang beolvasása
# --------------------------------------------------------------------------
def load_wav(path: str | Path, sr: int = SR, trim: bool = True) -> np.ndarray:
    """Beolvas egy wav fájlt, monóra kever, sr-re újramintavételez, csendet vág.

    A RAVDESS 48 kHz-es, tehát a 16 kHz pontosan a harmada — egész arányú
    újramintavételezéssel (resample_poly) gyors és pontos.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        y = resample_poly(y, sr // g, file_sr // g).astype(np.float32)
    if trim:
        y = trim_silence(y, sr)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, sr: int = SR, top_db: float = 35.0,
                 frame: int = 512, hop: int = 128) -> np.ndarray:
    """Egyszerű energiaalapú csendvágás a felvétel elejéről és végéről.

    A RAVDESS felvételek elején és végén fél-egy másodperc csend van; ez a
    jellemzőstatisztikákat (és az embeddinget is) fölöslegesen hígítja.
    """
    if len(y) < frame:
        return y
    n = 1 + (len(y) - frame) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + frame] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return y
    start = max(0, keep[0] * hop)
    end = min(len(y), keep[-1] * hop + frame)
    return y[start:end]


def load_all(paths, sr: int = SR, show_progress: bool = True) -> list[np.ndarray]:
    """Az összes felvétel beolvasása memóriába (1440 felvétel ≈ 350 MB float32)."""
    waves = []
    total = len(paths)
    for i, p in enumerate(paths):
        waves.append(load_wav(p, sr=sr))
        if show_progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{total} beolvasva")
    if show_progress:
        print(f"  kész: {total} felvétel")
    return waves


# --------------------------------------------------------------------------
# 3. Klasszikus (kézi) jellemzők
# --------------------------------------------------------------------------
def egemaps_features(waves, sr: int = SR, show_progress: bool = True):
    """eGeMAPSv02 jellemzők openSMILE-lal — 88 beszédszintű érték felvételenként.

    Ez az a szabványos jellemzőkészlet, amiről az előadáson szó volt.
    Ha az openSMILE nem érhető el, a librosa-alapú tartalékra esik vissza.
    """
    try:
        import opensmile
    except ImportError:
        print("Az openSMILE nem érhető el — a librosa-alapú jellemzőkre váltok.")
        return librosa_features(waves, sr=sr, show_progress=show_progress)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    rows, total = [], len(waves)
    for i, y in enumerate(waves):
        df = smile.process_signal(y.astype(np.float64), sr)
        rows.append(df.values[0])
        if show_progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{total} jellemzővektor")
    X = np.vstack(rows).astype(np.float32)
    names = list(smile.feature_names)
    if show_progress:
        print(f"  kész: {X.shape[0]} × {X.shape[1]} jellemzőmátrix (eGeMAPSv02)")
    return X, names


def librosa_features(waves, sr: int = SR, n_mfcc: int = 20,
                     show_progress: bool = True):
    """Tartalék jellemzőkészlet librosával: MFCC + delta + spektrális statisztikák.

    Alacsonyszintű jellemzőket számolunk keretenként, majd statisztikai
    függvényekkel (átlag, szórás) beszédszintű vektort készítünk belőlük —
    pontosan az a lépés, ami az előadáson „beszédszintű jellemzők” néven szerepelt.
    """
    import librosa
    rows, total = [], len(waves)
    for i, y in enumerate(waves):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        d1 = librosa.feature.delta(mfcc)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        cen = librosa.feature.spectral_centroid(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feats = []
        for block in (mfcc, d1, rms, zcr, cen, bw, roll):
            feats.append(block.mean(axis=1))
            feats.append(block.std(axis=1))
        rows.append(np.concatenate(feats))
        if show_progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{total} jellemzővektor")
    X = np.vstack(rows).astype(np.float32)
    names = ([f"mfcc{i}_mean" for i in range(n_mfcc)] + [f"mfcc{i}_std" for i in range(n_mfcc)]
             + [f"dmfcc{i}_mean" for i in range(n_mfcc)] + [f"dmfcc{i}_std" for i in range(n_mfcc)]
             + ["rms_mean", "rms_std", "zcr_mean", "zcr_std", "cen_mean", "cen_std",
                "bw_mean", "bw_std", "roll_mean", "roll_std"])
    if show_progress:
        print(f"  kész: {X.shape[0]} × {X.shape[1]} jellemzőmátrix (librosa)")
    return X, names


# --------------------------------------------------------------------------
# 4. Önfelügyelt beszédmodell-embedding
# --------------------------------------------------------------------------
def _is_oom(exc) -> bool:
    """Memóriahiány-e? Régebbi torch sima RuntimeError-t dob, nem OutOfMemoryError-t."""
    try:
        import torch
        if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


_MODEL_CACHE: dict = {}


def _load_speech_model(model_name: str, device: str):
    """A modellt csak egyszer töltjük be — a demónál ez sok várakozást spórol."""
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        from transformers import AutoFeatureExtractor, AutoModel
        fe = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval().to(device)
        _MODEL_CACHE[key] = (fe, model)
    return _MODEL_CACHE[key]


def wavlm_embeddings(waves, model_name: str = "microsoft/wavlm-base-plus",
                     sr: int = SR, batch_size: int | None = None,
                     layers: str | int | list = "all", device: str | None = None,
                     show_progress: bool = True):
    """Előtanított beszédmodell rejtett állapotainak időbeli átlaga felvételenként.

    layers:
        "all"  → (n_réteg+1, N, D) tömb: minden rejtett réteg külön
        int    → csak az adott réteg (0 = a konvolúciós enkóder kimenete)
        "mean" → az összes réteg átlaga

    A modell súlyait NEM hangoljuk: fagyasztott jellemzőkinyerőként használjuk.
    Ez az előadáson említett három adaptációs recept közül a legolcsóbb.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = suggested_batch_size()
    fe, model = _load_speech_model(model_name, device)

    out_layers = []
    total = len(waves)
    start = 0
    with torch.no_grad():
        while start < total:
            batch = waves[start:start + batch_size]
            try:
                enc = fe(batch, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = enc["input_values"].to(device)
                mask = enc.get("attention_mask")
                kwargs = {}
                if mask is not None:
                    mask = mask.to(device)
                    kwargs["attention_mask"] = mask
                hs = model(input_values, **kwargs).hidden_states  # (L+1) × (B,T,D)

                # időbeli átlag; ha van maszk, csak a valódi keretekre
                out_len = None
                if mask is not None:
                    try:
                        out_len = model._get_feat_extract_output_lengths(
                            mask.sum(-1)).cpu()
                    except Exception:
                        ratio = hs[0].shape[1] / input_values.shape[1]
                        out_len = (mask.sum(-1).cpu().float() * ratio).long()
                if out_len is None:
                    out_len = torch.full((input_values.shape[0],), hs[0].shape[1])
                out_len = out_len.clamp(min=1, max=hs[0].shape[1])
                pooled = []
                for h in hs:
                    T = h.shape[1]
                    idx = torch.arange(T, device=h.device)[None, :]
                    m = (idx < out_len.to(h.device)[:, None]).float()[:, :, None]
                    pooled.append(((h * m).sum(1) / m.sum(1).clamp(min=1))
                                  .cpu().numpy())
                out_layers.append(np.stack(pooled))  # (L+1, B, D)
                start += batch_size
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                # kevés VRAM (pl. 4 GB-os laptop GPU): felezzük a batch méretet
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise RuntimeError(
                        "Egyetlen felvétel sem fér a GPU memóriájába. Próbáld "
                        "device='cpu'-val, vagy rövidítsd a felvételeket.") from exc
                batch_size = max(1, batch_size // 2)
                print(f"\n  [kevés a VRAM — batch_size → {batch_size}]")
            if show_progress and start % (batch_size * 20) == 0:
                print(f"  {min(start, total)}/{total} embedding")

    E = np.concatenate(out_layers, axis=1).astype(np.float32)  # (L+1, N, D)
    if show_progress:
        print(f"  kész: {E.shape[0]} réteg × {E.shape[1]} felvétel × {E.shape[2]} dimenzió")
    if layers == "all":
        return E
    if layers == "mean":
        return E.mean(axis=0)
    return E[int(layers)]


# --------------------------------------------------------------------------
# 5. Felosztás
# --------------------------------------------------------------------------
def speaker_split(df, test_actors=(21, 22, 23, 24)):
    """Beszélőfüggetlen felosztás: a teszt színészei sosem szerepelnek tanításban."""
    test_mask = df["actor"].isin(list(test_actors)).values
    return ~test_mask, test_mask


def random_split(df, test_size: float = 0.2, seed: int = 0):
    """Véletlen, szegmensszintű felosztás — SZÁNDÉKOSAN HIBÁS referencia.

    A gyakorlaton ezt azért számoljuk ki, hogy lássuk, mennyivel optimistább
    eredményt ad, mint a beszélőfüggetlen felosztás.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                              stratify=df["emotion"].values)
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[tr] = True
    test_mask[te] = True
    return train_mask, test_mask


# --------------------------------------------------------------------------
# 6. Modell és kiértékelés
# --------------------------------------------------------------------------
def make_classifier(kind: str = "svm", C: float = 10.0):
    """Egységes osztályozó: standardizálás + SVM (vagy logisztikus regresszió)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    if kind == "svm":
        clf = SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced")
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    else:
        raise ValueError(f"ismeretlen osztályozó: {kind}")
    return make_pipeline(StandardScaler(), clf)


def run_experiment(X, y, train_mask, test_mask, kind: str = "svm",
                   name: str = "", verbose: bool = True):
    """Tanítás + kiértékelés egy lépésben. Visszaad egy szótárat az eredményekkel."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    clf = make_classifier(kind)
    clf.fit(X[train_mask], y[train_mask])
    pred = clf.predict(X[test_mask])
    true = y[test_mask]

    res = {
        "name": name,
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro"),
        "y_true": true,
        "y_pred": pred,
        "model": clf,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    if verbose:
        print(f"\n{name}")
        print(f"  tanító: {res['n_train']} · teszt: {res['n_test']}")
        print(f"  pontosság (accuracy): {res['accuracy']:.3f}")
        print(f"  makro-F1:             {res['macro_f1']:.3f}")
        print(classification_report(true, pred, zero_division=0))
    return res


def plot_confusion(res, labels=None, normalize: bool = True, ax=None,
                   title: str | None = None):
    """Tévesztési mátrix — soronként normalizálva a valós osztályra."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = labels or EMOTION_ORDER
    labels = [l for l in labels if l in set(res["y_true"]) | set(res["y_pred"])]
    M = confusion_matrix(res["y_true"], res["y_pred"], labels=labels)
    if normalize:
        M = M / M.sum(axis=1, keepdims=True).clip(min=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("becsült osztály")
    ax.set_ylabel("valós osztály")
    ax.set_title(title or f"{res['name']}\nmakro-F1 = {res['macro_f1']:.3f}",
                 fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() * 0.6 else "#444")
    return ax


def compare_results(results, title: str = "Összehasonlítás"):
    """Több kísérlet eredményének egymás mellé állítása oszlopdiagramon."""
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results]
    acc = [r["accuracy"] for r in results]
    f1 = [r["macro_f1"] for r in results]
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(1.9 * len(results) + 3, 4))
    ax.bar(x - 0.19, acc, 0.36, label="pontosság", color="#9DB9C9")
    ax.bar(x + 0.19, f1, 0.36, label="makro-F1", color="#1B4965")
    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax.text(xi - 0.19, a + .012, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, f + .012, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("teljesítmény")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    return ax


def plot_layer_scan(layer_scores, title: str = "Melyik réteg kódolja az érzelmet?"):
    """A rétegenkénti makro-F1 kirajzolása (lásd az előadás rétegsúlyozás-fóliáját)."""
    import matplotlib.pyplot as plt
    layers = sorted(layer_scores)
    vals = [layer_scores[l] for l in layers]
    best = int(np.argmax(vals))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(layers, vals, "o-", color="#1B4965", lw=1.8)
    ax.plot(layers[best], vals[best], "o", ms=12, mfc="none", mec="#A63A3A", mew=2)
    ax.annotate(f"legjobb: {layers[best]}. réteg ({vals[best]:.3f})",
                xy=(layers[best], vals[best]),
                xytext=(layers[best], vals[best] - 0.07),
                ha="center", color="#A63A3A", fontsize=9.5)
    ax.set_xlabel("rejtett réteg sorszáma (0 = konvolúciós enkóder kimenete)")
    ax.set_ylabel("makro-F1")
    ax.set_title(title)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    return ax


# --------------------------------------------------------------------------
# 7. Megjelenítés
# --------------------------------------------------------------------------
def plot_waveform_spectrogram(y, sr: int = SR, title: str = ""):
    """Hullámforma és spektrogram egymás alatt — gyors ránézés egy felvételre."""
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    f, t, S = spectrogram(y, fs=sr, nperseg=512, noverlap=384)
    S_db = 10 * np.log10(S + 1e-10)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6], "hspace": .15})
    axes[0].plot(np.arange(len(y)) / sr, y, lw=.5, color="#1B4965")
    axes[0].set_ylabel("amplitúdó")
    axes[0].set_yticks([])
    axes[0].set_title(title, fontsize=11, loc="left")
    axes[1].pcolormesh(t, f / 1000, S_db, shading="auto", cmap="magma",
                       vmin=S_db.max() - 70, vmax=S_db.max())
    axes[1].set_ylabel("kHz")
    axes[1].set_xlabel("idő [s]")
    return axes


def play(y, sr: int = SR):
    """Lejátszható hangvezérlő a notebookban."""
    from IPython.display import Audio, display
    display(Audio(y, rate=sr))


# --------------------------------------------------------------------------
# 8. Mikrofonos felvétel Colabban
# --------------------------------------------------------------------------
_RECORD_JS = """
const sleep = ms => new Promise(r => setTimeout(r, ms));
var record = time => new Promise(async resolve => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  let chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = async () => {
    const blob = new Blob(chunks);
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  };
  recorder.start();
  await sleep(time);
  recorder.stop();
  stream.getTracks().forEach(t => t.stop());
});
"""


def record_audio(seconds: float = 4.0, out_path: str = "felvetel.wav",
                 sr: int = SR) -> str:
    """Felvétel a mikrofonból, mentés 16 kHz-es mono wav fájlba.

    Colabban a böngésző mikrofonját használja (engedélyt fog kérni), helyben
    pedig a `sounddevice` csomagot (alapértelmezett bemeneti eszköz).
    """
    if in_colab():
        return _record_colab(seconds, out_path, sr)
    return _record_local(seconds, out_path, sr)


def _record_local(seconds: float, out_path: str, sr: int) -> str:
    """Helyi felvétel sounddevice-szal (Windows, macOS, Linux)."""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "A helyi felvételhez a sounddevice csomag kell:\n"
            "    pip install sounddevice\n"
            "Alternatíva: vedd fel a hangot bármilyen programmal, mentsd wav-ba, "
            "és használd a load_wav() függvényt.")
    import soundfile as sf

    print(f"Felvétel {seconds:.0f} másodpercig — beszélj a mikrofonba…")
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    y = y[:, 0]
    sf.write(out_path, y, sr)
    print(f"Mentve: {out_path}")
    return out_path


def list_microphones() -> None:
    """Kilistázza a helyi bemeneti eszközöket — ha nem a jó mikrofon szól bele."""
    try:
        import sounddevice as sd
    except ImportError:
        print("A sounddevice nincs telepítve (pip install sounddevice).")
        return
    print("Bemeneti eszközök:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            jel = " ← alapértelmezett" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{jel}")
    print("\nMásik eszköz választása:  import sounddevice as sd; "
          "sd.default.device = (<index>, None)")


def _record_colab(seconds: float, out_path: str, sr: int) -> str:
    """Felvétel a böngésző mikrofonjából Colabban."""
    from base64 import b64decode
    from IPython.display import Javascript, display
    from google.colab.output import eval_js

    print(f"Felvétel {seconds:.0f} másodpercig — beszélj a mikrofonba…")
    display(Javascript(_RECORD_JS))
    data = eval_js(f"record({int(seconds * 1000)})")
    raw = b64decode(data.split(",")[1])
    tmp = "felvetel_nyers.webm"
    with open(tmp, "wb") as f:
        f.write(raw)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-ar", str(sr), "-ac", "1", out_path], check=True)
    os.remove(tmp)
    print(f"Mentve: {out_path}")
    return out_path


def predict_one(path_or_wave, model, embed_fn, sr: int = SR, top_k: int = 3):
    """Egyetlen felvétel osztályozása a betanított modellel.

    embed_fn: egy függvény, ami a hullámformák listájából jellemzőmátrixot ad
              (pl. lambda w: su.wavlm_embeddings(w, layers=6, show_progress=False))
    """
    if isinstance(path_or_wave, (str, Path)):
        y = load_wav(path_or_wave, sr=sr)
    else:
        y = np.asarray(path_or_wave, dtype=np.float32)
    X = embed_fn([y])
    pred = model.predict(X)[0]
    print(f"Becsült érzelem: {pred}")
    if hasattr(model, "decision_function"):
        try:
            classes = model.classes_
            scores = model.decision_function(X)[0]
            order = np.argsort(scores)[::-1][:top_k]
            print("Legvalószínűbb osztályok:")
            for i in order:
                print(f"  {classes[i]:<14s} {scores[i]:+.2f}")
        except Exception:
            pass
    return pred


# --------------------------------------------------------------------------
# 9. Öndiagnosztika
# --------------------------------------------------------------------------
def self_check(verbose: bool = True) -> bool:
    """Gyors ellenőrzés, hogy a modul ép-e és a függőségek megvannak-e.

    Ha az importnál `AttributeError: module 'ser_utils' has no attribute ...`
    hibát kaptál, akkor a fájl üres vagy csonka — futtasd ezt, és nézd meg,
    mit ír ki.
    """
    import importlib
    hianyzik = []
    for mod in ("numpy", "scipy", "sklearn", "soundfile", "matplotlib", "pandas"):
        try:
            importlib.import_module(mod)
        except ImportError:
            hianyzik.append(mod)
    kell = ["download_ravdess", "scan_ravdess", "load_wav", "load_all",
            "egemaps_features", "wavlm_embeddings", "speaker_split",
            "random_split", "run_experiment", "plot_confusion", "record_audio"]
    hianyzo_fv = [f for f in kell if not hasattr(sys.modules[__name__], f)]

    ok = not hianyzik and not hianyzo_fv
    if verbose:
        print(f"ser_utils v{__version__}")
        for nev in ("torch", "transformers", "datasets", "sklearn"):
            try:
                m = importlib.import_module(nev)
                print(f"  {nev:<13s} {getattr(m, '__version__', '?')}")
            except ImportError:
                pass
        print(f"  függvények: {len(kell) - len(hianyzo_fv)}/{len(kell)} megvan")
        if hianyzo_fv:
            print(f"  HIÁNYZÓ FÜGGVÉNY: {', '.join(hianyzo_fv)} — a fájl csonka, töltsd le újra")
        if hianyzik:
            print(f"  HIÁNYZÓ CSOMAG: {', '.join(hianyzik)} — futtasd az ensure_packages()-t")
        print("  " + gpu_info())
        print("  RENDBEN" if ok else "  JAVÍTANDÓ — lásd fent")
    return ok


# --------------------------------------------------------------------------
# 10. Kiegészítő blokkok: modellösszehasonlítás, LOSO, osztályösszevonás
# --------------------------------------------------------------------------
def best_layer_score(E, y, train_mask, test_mask, kind: str = "svm",
                     show_progress: bool = True):
    """Végigpásztázza az SSL-modell rétegeit, és visszaadja a legjobbat.

    E: (rétegek, felvételek, dimenzió) tömb a wavlm_embeddings(layers="all")-ból.
    Visszatér: (legjobb_réteg, {réteg: makro-F1})
    """
    scores = {}
    for L in range(E.shape[0]):
        r = run_experiment(E[L], y, train_mask, test_mask, kind=kind,
                           name=f"réteg {L}", verbose=False)
        scores[L] = r["macro_f1"]
        if show_progress:
            print(f"  réteg {L:2d}: makro-F1 = {r['macro_f1']:.3f}")
    best = max(scores, key=scores.get)
    return best, scores


def emotion2vec_embeddings(waves, sr: int = SR,
                           model_name: str = "iic/emotion2vec_plus_large",
                           show_progress: bool = True):
    """emotion2vec beágyazások — érzelem-specifikus önfelügyelt modell.

    FIGYELEM: ez nem transformers-modell, a FunASR csomag kell hozzá:
        pip install -U funasr modelscope

    Visszatér: (N, D) tömb, megnyilatkozásszintű beágyazásokkal.
    """
    try:
        from funasr import AutoModel as FunAutoModel
    except ImportError:
        raise RuntimeError(
            "Az emotion2vec a FunASR csomagot igényli:\n"
            "    pip install -U funasr modelscope\n"
            "Ez egy nagyobb függőség; ha nem akarod telepíteni, hagyd ki ezt a "
            "részt — a WavLM-összehasonlítás önmagában is válaszol a kérdésre.")
    import soundfile as sf
    import tempfile

    model = FunAutoModel(model=model_name, disable_update=True)
    vecs, total = [], len(waves)
    with tempfile.TemporaryDirectory() as tmp:
        for i, w in enumerate(waves):
            fn = os.path.join(tmp, "x.wav")
            sf.write(fn, w, sr)
            res = model.generate(fn, granularity="utterance",
                                 extract_embedding=True, disable_pbar=True)
            vecs.append(np.asarray(res[0]["feats"], dtype=np.float32).ravel())
            if show_progress and (i + 1) % 200 == 0:
                print(f"  {i + 1}/{total} embedding")
    X = np.vstack(vecs)
    if show_progress:
        print(f"  kész: {X.shape[0]} × {X.shape[1]} (emotion2vec)")
    return X


def leave_one_speaker_out(X, y, groups, kind: str = "svm",
                          show_progress: bool = True):
    """Leave-one-speaker-out keresztvalidáció.

    Minden beszélő egyszer teszthalmaz, a többi tanító. Visszatér egy
    DataFrame-mel: beszélőnként pontosság és makro-F1.

    Ez a legszigorúbb kiértékelés, amit egy ekkora korpuszon érdemes csinálni:
    nemcsak egy átlagot ad, hanem megmutatja a beszélők közti SZÓRÁST is.
    """
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    groups = np.asarray(groups)
    y = np.asarray(y)
    sorok, egyedi = [], np.unique(groups)
    for k, g in enumerate(egyedi, 1):
        te = groups == g
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_classifier(kind)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        sorok.append({
            "beszelo": g,
            "n_teszt": int(te.sum()),
            "accuracy": accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
        })
        if show_progress:
            print(f"  [{k}/{len(egyedi)}] beszélő {g}: "
                  f"makro-F1 = {sorok[-1]['macro_f1']:.3f}")
    return pd.DataFrame(sorok)


def plot_speaker_scores(scores, metric: str = "macro_f1", csoport=None,
                        title: str | None = None):
    """A LOSO-eredmény beszélőnként, átlaggal és szórássávval.

    csoport: opcionális sorozat (pl. nem) — ha megadod, színnel elkülöníti.
    """
    import matplotlib.pyplot as plt

    sorrend = np.argsort(scores[metric].values)      # a rendezés sorrendje
    df = scores.iloc[sorrend].reset_index(drop=True)
    ertek = df[metric].values
    atlag, szoras = ertek.mean(), ertek.std(ddof=1)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    cs = None
    if csoport is not None:
        csoport = np.asarray(csoport)
        if len(csoport) == len(df):
            cs = csoport[sorrend]                     # ugyanúgy átrendezve!
        else:
            raise ValueError(
                f"a csoport hossza ({len(csoport)}) nem egyezik a "
                f"beszélők számával ({len(df)})")
    if cs is not None:
        szinek = {k: c for k, c in zip(sorted(set(cs)), ["#1B4965", "#C9A227"])}
        for k in szinek:
            m = cs == k
            ax.bar(np.arange(len(df))[m], ertek[m], color=szinek[k], label=str(k))
        ax.legend(frameon=False, fontsize=9.5)
    else:
        ax.bar(np.arange(len(df)), ertek, color="#1B4965")

    ax.axhline(atlag, color="#A63A3A", lw=1.4)
    ax.axhspan(atlag - szoras, atlag + szoras, color="#A63A3A", alpha=.10)
    ax.text(len(df) - 0.4, atlag + 0.012,
            f"átlag {atlag:.3f} ± {szoras:.3f}", color="#A63A3A",
            ha="right", fontsize=9.5)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["beszelo"].astype(str), fontsize=8.5)
    ax.set_xlabel("beszélő (növekvő teljesítmény szerint)")
    ax.set_ylabel(metric)
    ax.set_title(title or "Leave-one-speaker-out: beszélőnkénti eredmény")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    return ax


def merge_labels(y, csoportok: dict, uj_nev: str | None = None):
    """Osztályok összevonása.

    csoportok: {"új címke": ["régi1", "régi2"], ...}
    A fel nem sorolt címkék változatlanul maradnak.

    Példa:
        y7 = su.merge_labels(y, {"semleges+nyugodt": ["semleges", "nyugodt"]})
    """
    y = np.asarray(y, dtype=object).copy()
    for uj, regiek in csoportok.items():
        for r in regiek:
            y[y == r] = uj
    return y.astype(str)


# --------------------------------------------------------------------------
# 11. Ál-dimenziós regresszió (demó): kategóriák vetítése a valencia-arousal síkra
# --------------------------------------------------------------------------
# FIGYELEM: a RAVDESS-ben NINCS valencia-arousal annotáció — csak kategória és
# kétfokozatú intenzitás. Az alábbi koordináták a körkörös (circumplex) modell
# szerinti SZÁRMAZTATOTT értékek, nem mérés. A blokk célja kizárólag az, hogy
# megmutassa, hogyan néz ki egy dimenziós regresszió; az itt kapott számok NEM
# hasonlíthatók össze valódi dimenziós korpuszokon (MSP-Podcast, IEMOCAP)
# mért eredményekkel.
VA_MAP = {
    "semleges":     (0.00,  0.00),
    "nyugodt":      (0.50, -0.60),
    "öröm":         (0.70,  0.45),
    "szomorúság":  (-0.62, -0.48),
    "harag":       (-0.58,  0.70),
    "félelem":     (-0.70,  0.52),
    "undor":       (-0.78,  0.18),
    "meglepettség": (0.02,  0.93),
}


def category_to_va(labels, mapping: dict | None = None):
    """Kategóriacímkékből (N, 2) alakú valencia-arousal céltömb.

    A leképezés származtatott, nem annotáció — lásd a VA_MAP megjegyzését.
    """
    mapping = mapping or VA_MAP
    labels = np.asarray(labels)
    ismeretlen = sorted(set(labels) - set(mapping))
    if ismeretlen:
        raise ValueError(f"nincs koordináta ezekhez: {ismeretlen}")
    return np.array([mapping[l] for l in labels], dtype=np.float32)


def ccc(y_true, y_pred) -> float:
    """Lin-féle konkordancia-korrelációs együttható (CCC).

    A Pearson-korrelációval szemben a torzítást és a skálakülönbséget is
    bünteti — ezért ez a dimenziós érzelembecslés standard metrikája.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    nevezo = vt + vp + (mt - mp) ** 2
    return float(2 * cov / nevezo) if nevezo > 0 else 0.0


def run_va_regression(X, labels, train_mask, test_mask, mapping: dict | None = None,
                      C: float = 10.0, verbose: bool = True):
    """Ál-dimenziós regresszió: valencia és arousal becslése külön modellel.

    Visszaad egy szótárat a becslésekkel és a metrikákkal (CCC, Pearson, RMSE).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from scipy.stats import pearsonr

    Y = category_to_va(labels, mapping)
    res = {"nev": "ál-VA regresszió", "y_true": Y[test_mask],
           "labels_test": np.asarray(labels)[test_mask], "modellek": {}}
    becslesek = []
    for j, dim in enumerate(["valencia", "arousal"]):
        reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, gamma="scale"))
        reg.fit(X[train_mask], Y[train_mask, j])
        pred = reg.predict(X[test_mask])
        becslesek.append(pred)
        igaz = Y[test_mask, j]
        res["modellek"][dim] = reg
        res[dim] = {
            "ccc": ccc(igaz, pred),
            "pearson": float(pearsonr(igaz, pred)[0]),
            "rmse": float(np.sqrt(((igaz - pred) ** 2).mean())),
        }
        if verbose:
            m = res[dim]
            print(f"  {dim:<9s} CCC = {m['ccc']:.3f} · r = {m['pearson']:.3f} "
                  f"· RMSE = {m['rmse']:.3f}")
    res["y_pred"] = np.stack(becslesek, axis=1)
    res["atlag_ccc"] = (res["valencia"]["ccc"] + res["arousal"]["ccc"]) / 2
    if verbose:
        print(f"  átlagos CCC = {res['atlag_ccc']:.3f}")
    return res


def va_to_category(pred_va, mapping: dict | None = None):
    """A becsült (valencia, arousal) pontokat a legközelebbi kategóriához rendeli.

    Ezzel mérhető, hogy a regresszió hordoz-e TÖBB információt, mint a közvetlen
    osztályozás — vagy csak ugyanazt kerülő úton.
    """
    mapping = mapping or VA_MAP
    nevek = list(mapping)
    horgonyok = np.array([mapping[n] for n in nevek], dtype=np.float32)
    d = ((np.asarray(pred_va)[:, None, :] - horgonyok[None, :, :]) ** 2).sum(-1)
    return np.array(nevek, dtype=object)[d.argmin(1)].astype(str)


def plot_va_regression(res, mapping: dict | None = None):
    """Két panel: a VA-sík a horgonyokkal és a becslésekkel, plusz dimenziónkénti szórásdiagram."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    mapping = mapping or VA_MAP
    P, T = res["y_pred"], res["y_true"]
    cimkek = res["labels_test"]
    nevek = list(mapping)
    szinek = plt.cm.tab10(np.linspace(0, 1, 10))[:len(nevek)]
    szin_map = {n: szinek[i] for i, n in enumerate(nevek)}

    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_subplot(1, 2, 1)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color="#6B7280", lw=1.1,
                        ls=(0, (4, 4))))
    ax.axhline(0, color="#6B7280", lw=.9)
    ax.axvline(0, color="#6B7280", lw=.9)
    for n in nevek:
        m = cimkek == n
        if m.sum():
            ax.scatter(P[m, 0], P[m, 1], s=14, alpha=.45, color=szin_map[n],
                       edgecolors="none")
        vx, vy = mapping[n]
        ax.scatter([vx], [vy], s=170, marker="X", color=szin_map[n],
                   edgecolors="white", linewidths=1.6, zorder=5)
        # a feliratot sugárirányban kifelé toljuk, hogy ne fedjék egymást
        r = (vx ** 2 + vy ** 2) ** 0.5
        if r < 1e-6:
            dx, dy = 0.0, 16.0
        else:
            dx, dy = 20 * vx / r, 20 * vy / r
        ax.annotate(n, (vx, vy), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="center", fontsize=9, color="#1A1A1A",
                    zorder=6)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("valencia")
    ax.set_ylabel("arousal")
    ax.set_title("X = a kategóriák származtatott helye · pont = a modell becslése",
                 fontsize=10.5, loc="left", color="#6B7280")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    for k, dim in enumerate(["valencia", "arousal"]):
        a = fig.add_subplot(2, 2, 2 * (k + 1))
        for n in nevek:
            m = cimkek == n
            if m.sum():
                a.scatter(T[m, k] + np.random.default_rng(k).normal(0, .012, m.sum()),
                          P[m, k], s=12, alpha=.45, color=szin_map[n],
                          edgecolors="none")
        lim = [-1.05, 1.05]
        a.plot(lim, lim, color="#A63A3A", lw=1.1, ls="--")
        a.set_xlim(*lim)
        a.set_ylim(*lim)
        a.set_ylabel("becsült")
        if k == 1:
            a.set_xlabel("származtatott cél")
        m = res[dim]
        a.set_title(f"{dim}:  CCC = {m['ccc']:.3f} · r = {m['pearson']:.3f} "
                    f"· RMSE = {m['rmse']:.3f}", fontsize=10, loc="left",
                    color="#6B7280")
        a.grid(alpha=.2)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# ==========================================================================
#  12. KÉPI MODALITÁS — arckifejezés-alapú felismerés
# ==========================================================================
# A FER2013-at a clip-benchmark webdataset-tükréből töltjük (nyilvános, nem
# kér regisztrációt). FIGYELEM: ennek a tükörnek a címkesorrendje ÁBÉCÉS, nem
# a Kaggle-féle eredeti sorrend — innen a lenti leképezés.
FER_CLASSES_EN = ["angry", "disgusted", "fearful", "happy",
                  "neutral", "sad", "surprised"]
FER_CLASSES_HU = ["harag", "undor", "félelem", "öröm",
                  "semleges", "szomorúság", "meglepettség"]
FER_DATASET = "clip-benchmark/wds_fer2013"


def load_fer2013(n_train: int = 4000, n_test: int = 1500, seed: int = 0,
                 dataset_name: str = FER_DATASET, show_progress: bool = True):
    """FER2013 betöltése és véletlen mintavétele (48x48 szürkeárnyalatos arcok).

    A teljes korpusz 28,7 ezer tanító és 7,2 ezer teszt kép; órán ennek a
    töredéke is bőven elég. Visszatér: (kepek_tr, y_tr, kepek_te, y_te),
    ahol a képek PIL-objektumok, a címkék magyar nevek.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    if show_progress:
        print("elérhető splitek:", {k: len(v) for k, v in ds.items()})

    def kep_oszlop(d):
        for k in ("jpg", "image", "png"):
            if k in d.column_names:
                return k
        raise RuntimeError(f"nem találom a képoszlopot: {d.column_names}")

    def cimke_oszlop(d):
        for k in ("cls", "label", "labels"):
            if k in d.column_names:
                return k
        raise RuntimeError(f"nem találom a címkeoszlopot: {d.column_names}")

    def minta(split, n):
        d = ds[split]
        n = min(n, len(d))
        d = d.shuffle(seed=seed).select(range(n))
        ik, ck = kep_oszlop(d), cimke_oszlop(d)
        kepek = [r[ik].convert("L") for r in d]
        cimkek = np.array([FER_CLASSES_HU[int(r[ck])] for r in d])
        return kepek, cimkek

    teszt_split = "test" if "test" in ds else "validation"
    kep_tr, y_tr = minta("train", n_train)
    kep_te, y_te = minta(teszt_split, n_test)
    if show_progress:
        print(f"betöltve: {len(kep_tr)} tanító + {len(kep_te)} teszt kép, "
              f"méret: {kep_tr[0].size}")
    return kep_tr, y_tr, kep_te, y_te


def plot_image_grid(images, labels, per_class: int = 4, class_names=None,
                    title: str = "Mintaképek osztályonként"):
    """Osztályonként néhány mintakép — hogy lássák, mivel dolgozik a modell."""
    import matplotlib.pyplot as plt

    class_names = class_names or FER_CLASSES_HU
    labels = np.asarray(labels)
    sorok = [c for c in class_names if (labels == c).sum() > 0]
    fig, axes = plt.subplots(len(sorok), per_class,
                             figsize=(per_class * 1.4, len(sorok) * 1.5))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(sorok):
        idx = np.where(labels == c)[0][:per_class]
        for j in range(per_class):
            a = axes[i, j]
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if j < len(idx):
                a.imshow(np.asarray(images[idx[j]]), cmap="gray")
            if j == 0:
                a.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def hog_features(images, show_progress: bool = True):
    """Klasszikus képi jellemzők: HOG (Histogram of Oriented Gradients).

    Ez az előadáson említett "megjelenés alapú, hisztogramos" megközelítés —
    a mélytanulás előtti képi érzelemfelismerés tipikus jellemzőkészlete.
    Ha a scikit-image nem érhető el, lecsökkentett nyers pixelekre esik vissza.
    """
    try:
        from skimage.feature import hog
    except ImportError:
        print("A scikit-image nem érhető el — lecsökkentett nyers pixelekre váltok.")
        X = np.stack([np.asarray(im.resize((24, 24)), dtype=np.float32).ravel() / 255
                      for im in images])
        return X, [f"px{i}" for i in range(X.shape[1])]

    sorok, total = [], len(images)
    for i, im in enumerate(images):
        a = np.asarray(im, dtype=np.float32) / 255.0
        sorok.append(hog(a, orientations=8, pixels_per_cell=(6, 6),
                         cells_per_block=(2, 2), feature_vector=True))
        if show_progress and (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{total} HOG-vektor")
    X = np.vstack(sorok).astype(np.float32)
    if show_progress:
        print(f"  kész: {X.shape[0]} x {X.shape[1]} HOG-jellemzőmátrix")
    return X, [f"hog{i}" for i in range(X.shape[1])]


def _clip_features(model, out, kind: str):
    """A CLIP jellemzőtenzor kinyerése, transformers-verziótól függetlenül.

    A transformers 4.x-ben a get_image_features / get_text_features kész
    tenzort ad vissza. Az 5.x-ben egy BaseModelOutputWithPooling objektumba
    csomagolja, ahol a `pooler_output` MÁR a projektált jellemző — ilyenkor
    csak ki kell bontani.

    Biztonság kedvéért megnézzük a dimenziót is: a vetítést csak akkor
    alkalmazzuk, ha a kapott vektor mérete egyértelműen a vetítés BEMENETI
    mérete (és az eltér a kimenetitől). Így sem a már projektált, sem a nyers
    kimenet nem romlik el.
    """
    import torch

    if torch.is_tensor(out):
        return out

    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = getattr(out, "image_embeds", None) or getattr(out, "text_embeds", None)
    if pooled is None:
        rejtett = getattr(out, "last_hidden_state", None)
        if rejtett is not None:
            pooled = rejtett[:, 0]
        elif isinstance(out, (tuple, list)) and len(out) > 1:
            pooled = out[1]
        else:
            raise RuntimeError(
                f"nem tudom kinyerni a CLIP jellemzőket ebből: {type(out).__name__}")

    proj = getattr(model, "visual_projection" if kind == "image"
                   else "text_projection", None)
    be = getattr(proj, "in_features", None)
    ki = getattr(proj, "out_features", None)
    if proj is not None and be is not None and ki is not None \
            and pooled.shape[-1] == be and be != ki:
        # nyers, vetítés előtti kimenet — pótoljuk a vetítést
        pooled = proj(pooled)
    return pooled


_CLIP_CACHE: dict = {}


def _load_clip(model_name: str, device: str):
    if (model_name, device) not in _CLIP_CACHE:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_name).eval().to(device)
        proc = CLIPProcessor.from_pretrained(model_name)
        _CLIP_CACHE[(model_name, device)] = (model, proc)
    return _CLIP_CACHE[(model_name, device)]


def clip_image_embeddings(images, model_name: str = "openai/clip-vit-base-patch32",
                          batch_size: int | None = None, device: str | None = None,
                          show_progress: bool = True):
    """CLIP képi beágyazások — fagyasztott vizuális alapmodell jellemzőkinyerőként.

    Pontosan ugyanaz a recept, mint a hangnál a WavLM: nem hangolunk, csak
    reprezentációt veszünk ki, és kis fejet tanítunk rá.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = max(8, suggested_batch_size(64))
    model, proc = _load_clip(model_name, device)

    vektorok, total, start = [], len(images), 0
    with torch.no_grad():
        while start < total:
            batch = [im.convert("RGB") for im in images[start:start + batch_size]]
            try:
                be = proc(images=batch, return_tensors="pt").to(device)
                f = _clip_features(model, model.get_image_features(**be), "image")
                f = f / f.norm(dim=-1, keepdim=True)
                vektorok.append(f.cpu().numpy())
                start += batch_size
                if show_progress and start % (batch_size * 10) == 0:
                    print(f"  {min(start, total)}/{total} képi embedding")
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise RuntimeError(
                        "Egyetlen kép sem fér a GPU memóriájába. Próbáld "
                        "device='cpu'-val.") from exc
                batch_size = max(1, batch_size // 2)
                print(f"  [kevés a VRAM — batch_size -> {batch_size}]")
    X = np.vstack(vektorok).astype(np.float32)
    if show_progress:
        print(f"  kész: {X.shape[0]} x {X.shape[1]} (CLIP)")
    return X


def clip_zeroshot(images, class_names=None, sablon: str = "a photo of a {} face",
                  model_name: str = "openai/clip-vit-base-patch32",
                  device: str | None = None, batch_size: int = 64):
    """CLIP zero-shot osztályozás: tanítás NÉLKÜL, pusztán szöveges leírásokkal.

    A kategórianeveket angolul adjuk meg (a CLIP angol szövegen tanult),
    a visszaadott címkék viszont a magyar nevek.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = class_names or FER_CLASSES_EN
    model, proc = _load_clip(model_name, device)

    with torch.no_grad():
        szoveg = proc(text=[sablon.format(c) for c in class_names],
                      return_tensors="pt", padding=True).to(device)
        T = _clip_features(model, model.get_text_features(**szoveg), "text")
        T = T / T.norm(dim=-1, keepdim=True)

        becslesek = []
        ellenorizve = False
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[start:start + batch_size]]
            be = proc(images=batch, return_tensors="pt").to(device)
            I = _clip_features(model, model.get_image_features(**be), "image")
            I = I / I.norm(dim=-1, keepdim=True)
            if not ellenorizve:
                if I.shape[-1] != T.shape[-1]:
                    raise RuntimeError(
                        f"a képi ({I.shape[-1]}) és a szöveges ({T.shape[-1]}) "
                        "jellemzők dimenziója eltér — a zero-shot így "
                        "értelmetlen lenne. Frissítsd a ser_utils.py-t.")
                ellenorizve = True
            becslesek.append((I @ T.T).argmax(dim=-1).cpu().numpy())
    idx = np.concatenate(becslesek)
    return np.array(FER_CLASSES_HU, dtype=object)[idx].astype(str)


# ==========================================================================
#  13. SZÖVEGES MODALITÁS — szentiment magyar nyelven
# ==========================================================================
HUSST_DATASET = "NYTK/HuSST"
HUSST_HU = {"negative": "negatív", "neutral": "semleges", "positive": "pozitív"}


def load_husst(dataset_name: str = HUSST_DATASET, show_progress: bool = True):
    """HuSST: a Stanford Sentiment Treebank magyar változata (NYTK).

    A hivatalos teszt split címkézetlen, ezért a validációs halmazt használjuk
    teszthalmazként. Visszatér: (szoveg_tr, y_tr, szoveg_te, y_te).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    if show_progress:
        print("elérhető splitek:", {k: len(v) for k, v in ds.items()})

    def oszlop(d, jeloltek):
        for k in jeloltek:
            if k in d.column_names:
                return k
        raise RuntimeError(f"nem találom az oszlopot {jeloltek} közül: {d.column_names}")

    def kinyer(split):
        d = ds[split]
        sk = oszlop(d, ["sentence", "Sent", "text"])
        ck = oszlop(d, ["label", "Label", "labels"])
        szovegek = [r[sk] for r in d]
        cimkek = [HUSST_HU.get(str(r[ck]).lower(), str(r[ck])) for r in d]
        return szovegek, np.array(cimkek)

    teszt = "validation" if "validation" in ds else "test"
    sz_tr, y_tr = kinyer("train")
    sz_te, y_te = kinyer(teszt)
    if show_progress:
        print(f"betöltve: {len(sz_tr)} tanító + {len(sz_te)} teszt mondat "
              f"(teszthalmaz: a(z) '{teszt}' split, mert a hivatalos test "
              f"címkézetlen)")
    return sz_tr, y_tr, sz_te, y_te


def tfidf_pipeline(C: float = 4.0, ngram=(1, 2), min_df: int = 2,
                   max_features: int | None = 50_000):
    """TF-IDF + logisztikus regresszió — a klasszikus szövegosztályozó alapvonal."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(ngram_range=ngram, min_df=min_df,
                        max_features=max_features, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
    )


def tfidf_top_features(pipe, k: int = 12):
    """A legerősebb súlyú szavak osztályonként.

    Ez a klasszikus megközelítés egyik fő előnye: a modell megmagyarázható.
    """
    import pandas as pd

    vec = pipe.named_steps["tfidfvectorizer"]
    clf = pipe.named_steps["logisticregression"]
    nevek = np.array(vec.get_feature_names_out())
    sorok = {}
    for i, osztaly in enumerate(clf.classes_):
        w = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        legjobb = np.argsort(w)[::-1][:k]
        sorok[str(osztaly)] = [f"{nevek[j]} ({w[j]:+.2f})" for j in legjobb]
    return pd.DataFrame(sorok)


_TEXT_CACHE: dict = {}


def text_embeddings(texts, model_name: str =
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    batch_size: int = 64, device: str | None = None,
                    max_length: int = 128, show_progress: bool = True):
    """Mondatbeágyazások előtanított nyelvi modellel, maszkolt átlagolással.

    Ugyanaz a recept, mint a hangnál: fagyasztott modell + átlagolás.
    Magyar modellel is működik, pl. model_name="SZTAKI-HLT/hubert-base-cc".
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (model_name, device) not in _TEXT_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModel.from_pretrained(model_name).eval().to(device)
        _TEXT_CACHE[(model_name, device)] = (tok, mod)
    tok, mod = _TEXT_CACHE[(model_name, device)]

    vektorok, total = [], len(texts)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = list(texts[start:start + batch_size])
            be = tok(batch, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt").to(device)
            ki = mod(**be).last_hidden_state
            m = be["attention_mask"].unsqueeze(-1).float()
            atlag = (ki * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vektorok.append(atlag.cpu().numpy())
            if show_progress and (start // batch_size) % 20 == 19:
                print(f"  {min(start + batch_size, total)}/{total} mondat")
    X = np.vstack(vektorok).astype(np.float32)
    if show_progress:
        print(f"  kész: {X.shape[0]} x {X.shape[1]} ({model_name.split('/')[-1]})")
    return X


def predict_texts(mondatok, model=None, embed_fn=None, pipe=None):
    """Saját mondatok osztályozása — a demóhoz.

    Vagy egy TF-IDF pipeline-t adj át (pipe), vagy egy betanított osztályozót
    és a hozzá tartozó beágyazó függvényt (model + embed_fn).
    """
    import pandas as pd

    mondatok = list(mondatok)
    if pipe is not None:
        pred = pipe.predict(mondatok)
    else:
        pred = model.predict(embed_fn(mondatok))
    return pd.DataFrame({"mondat": mondatok, "becslés": pred})
