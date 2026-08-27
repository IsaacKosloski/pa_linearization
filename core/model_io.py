"""
core/model_io.py

Serialização e desserialização de modelos NARX (interceptos + A1 + A2 + betas).

Filosofia
---------
O GA otimiza apenas os `betas`, mas um artefato "cabal" deve ser
AUTO-SUFICIENTE e REPRODUZÍVEL. Por isso salvamos o NARXModelConfig
COMPLETO, não só o vetor de betas, junto de metadados de proveniência
(RMSE/NMSE, hiperparâmetros do GA, seed, dataset, versões de libs).

Formatos
--------
  - JSON  (primário): portável, versionável em git, legível por humanos.
  - TXT   (secundário): mesmo estilo de Previous_PA_Model.txt, para diff.
  - NPZ   (opcional): binário numérico puro (arrays), para reuso rápido.

Cuidado com numpy
-----------------
Após o GA, os betas são reconstruídos via ExogenousBetas.from_array(),
o que produz np.float64. json.dump não serializa np.float64 nativamente,
então usamos um conversor (_to_builtin).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core.model_config import (
    NARXModelConfig,
    Intercepts,
    AutoregressiveMatrix,
    ExogenousBetas,
)

FORMAT_VERSION = 1


# ─────────────────────────────────────────────────────────────
#  Conversão config <-> dict
# ─────────────────────────────────────────────────────────────
def _to_builtin(o):
    """Converte tipos numpy em tipos nativos para o json.dump."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo não serializável: {type(o)}")


def config_to_dict(config: NARXModelConfig) -> dict:
    """asdict recorre nos dataclasses aninhados → dict puro de floats."""
    return asdict(config)


def config_from_dict(d: dict) -> NARXModelConfig:
    """Reconstrói o NARXModelConfig a partir do dict aninhado."""
    return NARXModelConfig(
        intercepts=Intercepts(**d["intercepts"]),
        A1=AutoregressiveMatrix(**d["A1"]),
        A2=AutoregressiveMatrix(**d["A2"]),
        betas=ExogenousBetas(**d["betas"]),
    )


# ─────────────────────────────────────────────────────────────
#  JSON (formato primário)
# ─────────────────────────────────────────────────────────────
def save_model_json(
    config: NARXModelConfig,
    path: str | Path,
    *,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Salva o modelo completo em JSON, com envelope de metadados.

    :param config:   NARXModelConfig otimizado (best_config do GA)
    :param path:     caminho do .json de saída
    :param metadata: dict livre com RMSE, NMSE, GAConfig, seed, dataset...
    :return:         Path do arquivo escrito
    """
    payload = {
        "format_version": FORMAT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "metadata": metadata or {},
        "model": config_to_dict(config),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_to_builtin)

    return path


def load_model_json(path: str | Path) -> Tuple[NARXModelConfig, dict]:
    """
    Carrega um modelo salvo em JSON.

    :return: (config, metadata)
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("format_version") != FORMAT_VERSION:
        # Não é erro fatal: apenas avisa. Ajuste se quebrar compatibilidade.
        print(
            f"[model_io] Aviso: format_version={payload.get('format_version')} "
            f"≠ esperado {FORMAT_VERSION}."
        )

    config = config_from_dict(payload["model"])
    return config, payload.get("metadata", {})


# ─────────────────────────────────────────────────────────────
#  TXT (estilo Previous_PA_Model.txt)
# ─────────────────────────────────────────────────────────────
def _beta_key(field_name: str) -> str:
    """
    Converte o nome do atributo do dataclass no rótulo do TXT.
    Ex.: 'Xreal_lag1_deg2_Yreal' -> 'beta.Xreal_lag1_deg2.Yreal'
    """
    if field_name.endswith("_Yreal"):
        return f"beta.{field_name[:-6]}.Yreal"
    if field_name.endswith("_Yimg"):
        return f"beta.{field_name[:-5]}.Yimg"
    return f"beta.{field_name}"


def save_model_txt(config: NARXModelConfig, path: str | Path) -> Path:
    """
    Dump legível no mesmo estilo de Previous_PA_Model.txt.
    Útil para versionar e dar `git diff` entre modelos.

    A ordem dos betas segue a definição do dataclass (agrupada por saída),
    que difere ligeiramente do arquivo original, mas é completa e estável.
    """
    ic = config.intercepts
    a1 = config.A1
    a2 = config.A2

    lines = []
    lines.append("===== INTERCEPTOS =====")
    lines.append(f"intercept.Yreal   {ic.Yreal:.6f}")
    lines.append(f"intercept.Yimg    {ic.Yimg:.6f}")
    lines.append("")

    lines.append("===== MATRIZ A_1 =====")
    lines.append(f"Yreal(t-1) -> Yreal(t) = {a1.Yreal_to_Yreal:.10f}")
    lines.append(f"Yimg(t-1) -> Yreal(t) = {a1.Yimg_to_Yreal:.10f}")
    lines.append(f"Yreal(t-1) -> Yimg(t) = {a1.Yreal_to_Yimg:.10f}")
    lines.append(f"Yimg(t-1) -> Yimg(t) = {a1.Yimg_to_Yimg:.10f}")
    lines.append("")

    lines.append("===== MATRIZ A_2 =====")
    lines.append(f"Yreal(t-2) -> Yreal(t) = {a2.Yreal_to_Yreal:.10f}")
    lines.append(f"Yimg(t-2) -> Yreal(t) = {a2.Yimg_to_Yreal:.10f}")
    lines.append(f"Yreal(t-2) -> Yimg(t) = {a2.Yreal_to_Yimg:.10f}")
    lines.append(f"Yimg(t-2) -> Yimg(t) = {a2.Yimg_to_Yimg:.10f}")
    lines.append("")

    lines.append("===== VARIÁVEIS EXÓGENAS =====")
    for name, value in config.betas.to_dict().items():
        lines.append(f"{_beta_key(name)} = {float(value):.10f}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────
#  NPZ (binário numérico opcional)
# ─────────────────────────────────────────────────────────────
def save_model_npz(config: NARXModelConfig, path: str | Path) -> Path:
    """Salva arrays crus (rápido de recarregar em pipelines numéricos)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        intercepts=config.intercepts.to_array(),
        A1=config.A1.to_matrix(),
        A2=config.A2.to_matrix(),
        betas=config.betas.to_array(),
    )
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


# ─────────────────────────────────────────────────────────────
#  Verificação (round-trip)
# ─────────────────────────────────────────────────────────────
def verify_roundtrip(
    config: NARXModelConfig,
    X: np.ndarray,
    Y_true: np.ndarray,
    json_path: str | Path,
    *,
    atol: float = 1e-9,
) -> bool:
    """
    Salva → recarrega → reexecuta a predição e confere se o RMSE bate.
    É o teste que garante que o artefato salvo é fiel ao modelo em memória.
    """
    from core.narx_engine import NARXEngine

    engine_orig = NARXEngine(config)
    rmse_orig = NARXEngine.rmse(Y_true, engine_orig.predict(X))

    reloaded, _ = load_model_json(json_path)
    engine_reload = NARXEngine(reloaded)
    rmse_reload = NARXEngine.rmse(Y_true, engine_reload.predict(X))

    ok = abs(rmse_orig - rmse_reload) <= atol
    status = "OK" if ok else "FALHOU"
    print(
        f"[model_io] Verificação round-trip: {status} | "
        f"RMSE orig={rmse_orig:.10f} recarregado={rmse_reload:.10f}"
    )
    return ok