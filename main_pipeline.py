"""
Pipeline principal de Otimização do modelo do PA.

Orquestra:
    1. Carregamento / geração dos dados
    2. Inferência com o modelo semente
    3. Execução do Algoritmo Genético
    4. Geração dos gráficos de análise

Execute com:
    python main_pipeline.py

Dependências:
    pip install deap numpy pandas matplotlib scipy
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict

# --- Módulos do projeto
from core.model_config  import NARXModelConfig
from core.narx_engine   import NARXEngine, compute_gain
from ga.genetic_algorithm import PALinearizationGA, GAConfig
from core.model_io import save_model_json, save_model_txt, save_model_npz, verify_roundtrip
from visualization.plots  import (
    plot_constellation,
    plot_am_am_pm,
    plot_ga_convergence,
    plot_full_dashboard,
)



# --- Utilitário de Diretórios
def create_run_directory(base_dir: str = "./output", prefix: str = "output") -> Path:
    """
    Cria um novo subdiretório numerado dentro de `base_dir`.

    Procura subpastas já existentes no formato {prefix}NN (ex.: output01,
    output02, ...), descobre o maior número usado e cria o próximo da
    sequência. Assim, cada execução do pipeline gera uma pasta nova e nunca
    sobrescreve os resultados anteriores.

    Exemplo:
        Se já existem  output/output01  e  output/output02,
        a função cria e retorna  output/output03.

    :param base_dir : pasta-mãe que guarda todas as execuções
    :param prefix   : prefixo do nome de cada subpasta
    :return         : Path para o subdiretório recém-criado
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)   # garante que a pasta-mãe existe

    # Regex que casa apenas nomes como "output07" e captura o número (07)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    existing_nums = []
    for item in base.iterdir():               # percorre o conteúdo de base/
        if item.is_dir():                     # só interessam pastas
            match = pattern.match(item.name)
            if match:
                existing_nums.append(int(match.group(1)))

    # Se nenhuma pasta existe ainda, default=0 garante que o próximo seja 1
    next_num = max(existing_nums, default=0) + 1
    run_dir = base / f"{prefix}{next_num:02d}"   # :02d = 2 dígitos com zero à esquerda
    run_dir.mkdir()
    return run_dir

# --- Utilitário de Dados
def load_or_generate_data(
    csv_path: str | None = None,
    n_samples: int = 4096,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados de um CSV ou gera dados sintéticos para teste.

    Para dados reais, o CSV deve ter as colunas:
        Xreal, Ximg, Yreal, Yimg

    Dados sintéticos simulam um sinal OFDM típico passando por um PA com não-linearidade de 3ª ordem.

    :returns
    X : np.ndarray complex (N,) — sinal de entrada
    Y : np.ndarray complex (N,) — sinal de saída medido
    """
    if csv_path and Path(csv_path).exists():
        print(f"Carregando dados de: {csv_path}")
        df = pd.read_csv(csv_path)
        X = (df["Xreal"] + 1j * df["Ximg"]).to_numpy()
        Y = (df["Yreal"] + 1j * df["Yimg"]).to_numpy()
        print(f"   {len(X)} amostras carregadas.")
        return X, Y

    print(f"Gerando dados sintéticos ({n_samples} amostras)...")
    rng = np.random.default_rng(seed)

    # Simula envoltória complexa de um sinal OFDM de 64 subportadoras
    t = np.arange(n_samples)
    X = np.zeros(n_samples, dtype=complex)
    for _ in range(64):
        freq  = rng.uniform(0, 0.5)
        phase = rng.uniform(0, 2 * np.pi)
        X += (rng.standard_normal() + 1j * rng.standard_normal()) * \
             np.exp(1j * (2 * np.pi * freq * t + phase))
    X /= np.max(np.abs(X))  # Normaliza para amplitude máxima = 1

    # Ganho nominal do PA (≈ 23.45 do modelo)
    G_nominal = 23.45 + 1.47j

    # Adiciona não-linearidade de Saleh (compressão de ganho)
    amp = np.abs(X)
    G_nl = G_nominal / (1 + 0.3 * amp**2)   # AM-AM
    phi_nl = np.exp(1j * 0.1 * amp**2)       # AM-PM

    Y = G_nl * phi_nl * X
    Y += 0.01 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))

    return X, Y

def build_dataset(
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred_seed: np.ndarray,
    Y_pred_opt:  np.ndarray,
    G_linearized: np.ndarray,
) -> pd.DataFrame:
    """
    Constrói o DataFrame final com todas as colunas relevantes.

    Colunas geradas:
        Xreal, Ximg          → entrada complexa
        Yreal, Yimg          → saída medida (ground truth)
        Ypred_seed_real/img  → predição do modelo semente
        Ypred_opt_real/img   → predição do modelo otimizado
        G_real, G_img        → ganho complexo G[n] = Y_opt[n]/X[n]
        G_magnitude_dB       → |G[n]| em dB
        G_phase_deg          → ∠G[n] em graus
        X_G_Yreal            → Re(X[n] * G_media) — coluna "X*G=Y"
        X_G_Yimg             → Im(X[n] * G_media)
    """
    # Ganho médio (escalar complexo) para a coluna X[n]*G=Y[n]
    G_mean = np.nanmean(G_linearized[np.abs(G_linearized) > 0])

    df = pd.DataFrame({
        # Entrada
        "Xreal":              X.real,
        "Ximg":               X.imag,
        "X_magnitude":        np.abs(X),
        # Saída medida
        "Yreal":              Y_true.real,
        "Yimg":               Y_true.imag,
        # Predição semente (modelo original)
        "Ypred_seed_real":    Y_pred_seed.real,
        "Ypred_seed_img":     Y_pred_seed.imag,
        "RMSE_seed_point":    np.abs(Y_true - Y_pred_seed),
        # Predição otimizada (GA)
        "Ypred_opt_real":     Y_pred_opt.real,
        "Ypred_opt_img":      Y_pred_opt.imag,
        "RMSE_opt_point":     np.abs(Y_true - Y_pred_opt),
        # Ganho complexo por amostra
        "G_real":             G_linearized.real,
        "G_img":              G_linearized.imag,
        "G_magnitude":        np.abs(G_linearized),
        "G_magnitude_dB":     20 * np.log10(np.abs(G_linearized) + 1e-15),
        "G_phase_deg":        np.angle(G_linearized, deg=True),
        # -- Coluna ganho linearizado: X[n] * G = Y[n]
        # Multiplica cada amostra pelo ganho médio linearizado
        "X_times_G_real":     (X * G_mean).real,
        "X_times_G_img":      (X * G_mean).imag,
        "X_times_G_magnitude": np.abs(X * G_mean),
    })

    return df

# --- Pipeline Principal
def run_pipeline(
    csv_path:       str | None = None,
    n_samples:      int   = 4096,
    n_generations:  int   = 100,
    population_size:int   = 50,
    output_dir:     str   = "./output",
    show_plots:     bool  = True,
) -> pd.DataFrame:
    """
    Executa o pipeline completo de linearização do PA.

    :parameter
    csv_path       : caminho para CSV com dados reais (None = dados sintéticos)
    n_samples      : amostras para dados sintéticos
    n_generations  : gerações do GA
    population_size: população do GA
    output_dir     : pasta para salvar outputs
    show_plots     : exibe os gráficos ao final

    :return
    df : pd.DataFrame com todas as colunas incluindo X[n]*G=Y[n]
    """

    out = create_run_directory(output_dir)
    print(f"Resultados desta execução em: {out}")

    # 1 - Dados
    X, Y_true = load_or_generate_data(csv_path, n_samples)

    # 2 - Modelo semente
    seed_config = NARXModelConfig()  # carrega os coeficientes do model_config.py
    seed_engine = NARXEngine(seed_config)
    Y_pred_seed = seed_engine.predict(X)

    rmse_seed = NARXEngine.rmse(Y_true, Y_pred_seed)
    nmse_seed = NARXEngine.nmse(Y_true, Y_pred_seed)
    print(f"   RMSE semente : {rmse_seed:.6f}")
    print(f"   NMSE semente : {nmse_seed:.2f} dB")
    print(f"   Ganho linear : {seed_config.gain_linear_dB:.2f} dB")

    # 3 - Algoritmo Genético
    ga_config = GAConfig(
        population_size     = population_size,
        n_generations       = n_generations,
        crossover_prob      = 0.70,
        mutation_prob       = 0.20,
        mutation_sigma_rel  = 0.05,
        tournament_size     = 3,
        n_elites            = 2,
        random_seed         = 42,
        init_perturbation   = 0.10,
    )

    ga = PALinearizationGA(
        seed_config = seed_config,
        X_input     = X,
        Y_true      = Y_true,
        ga_config   = ga_config,
    )

    best_config, logbook = ga.run()

    # 4 - Predição com modelo otimizado
    print("\nPredizendo com modelo otimizado...")
    opt_engine = NARXEngine(best_config)
    Y_pred_opt = opt_engine.predict(X)

    rmse_opt = NARXEngine.rmse(Y_true, Y_pred_opt)
    nmse_opt = NARXEngine.nmse(Y_true, Y_pred_opt)
    print(f"   RMSE otimizado : {rmse_opt:.6f}  (redução: {rmse_seed - rmse_opt:.6f})")
    print(f"   NMSE otimizado : {nmse_opt:.2f} dB")
    print(f"   Ganho linear   : {best_config.gain_linear_dB:.2f} dB")

    # ── Salvar modelo otimizado ───────────────────────────────
    print("\nSalvando modelo otimizado...")

    metadata = {
        "rmse_seed": float(rmse_seed),
        "rmse_opt": float(rmse_opt),
        "nmse_opt_dB": float(nmse_opt),
        "gain_linear_dB": float(best_config.gain_linear_dB),
        "ga_config": asdict(ga_config),  # todos os hiperparâmetros do GA
        "random_seed": ga_config.random_seed,
        "n_samples": int(len(X)),
        "dataset": str(csv_path),
        "n_generations": int(n_generations),
    }

    json_path = out / "modelo_otimizado.json"
    save_model_json(best_config, json_path, metadata=metadata)
    save_model_txt(best_config, out / "modelo_otimizado.txt")
    save_model_npz(best_config, out / "modelo_otimizado.npz")
    print(f"   JSON: {json_path}")

    # Confere que o arquivo salvo reproduz o RMSE em memória
    verify_roundtrip(best_config, X, Y_true, json_path)

    # 5 - Ganho linearizado G[n]
    G_linearized = compute_gain(X, Y_pred_opt)
    G_mean = np.nanmean(G_linearized[np.abs(G_linearized) > 0])
    print(f"\n Ganho médio linearizado:")
    print(f"   |G| = {abs(G_mean):.4f}  ({20 * np.log10(abs(G_mean)):.2f} dB)")
    print(f"   ∠G  = {np.angle(G_mean, deg=True):.2f}°")

    # 6 - DataFrame final
    print("\n Construindo DataFrame")
    df = build_dataset(X, Y_true, Y_pred_seed, Y_pred_opt, G_linearized)

    csv_out = out / "pa_linearization_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"   Salvo em: {csv_out}")
    print(f"   Shape: {df.shape}")
    print(f"\n   Colunas: {list(df.columns)}")

    # 8 - Gráficos
    print("\nGerando gráficos...")

    fig_const = plot_constellation(
        X, Y_pred_seed, Y_pred_opt,
        title_suffix="— Linearização GA",
        save_path=str(out / "constellation.png"),
    )
    fig_amam = plot_am_am_pm(
        X, Y_pred_seed, Y_pred_opt,
        save_path=str(out / "am_am_pm.png"),
    )
    fig_ga = plot_ga_convergence(
        logbook,
        save_path=str(out / "ga_convergence.png"),
    )
    fig_dash = plot_full_dashboard(
        X, Y_pred_seed, Y_pred_opt,
        logbook=logbook,
        save_path=str(out / "dashboard.png"),
    )

    if show_plots:
        plt.show()

    print("\nPipeline finalizado")
    return df

# Main
if __name__ == "__main__":
    df = run_pipeline(
        csv_path        = None,
        n_samples       = 4096,          # usado apenas para dados sintéticos
        n_generations   = 100,
        population_size = 50,
        output_dir      = "./output",
        show_plots      = True,
    )

    # Inspeciona as primeiras linhas
    print("\nPrimeiras 5 linhas do DataFrame:")
    print(df[["Xreal", "Ximg", "Yreal", "Yimg",
              "Ypred_opt_real", "Ypred_opt_img",
              "G_magnitude_dB", "G_phase_deg",
              "X_times_G_real", "X_times_G_img"]].head())
