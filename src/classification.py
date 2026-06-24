import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from pathlib import Path
from .preprocessing import resize_rgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

logger = logging.getLogger(__name__)
SEED = 42


def run_classification_pipeline():
    """
    Executa todo o pipeline de classificação: carregamento de features,
    padronização, treinamento de KNN e SVM, avaliação e visualizações.

    Parâmetros
    ----------
    Nenhum

    Retorna
    -------
    Nenhum
    """

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    logger.info("=== CARREGANDO FEATURES ===")
    data = np.load("outputs/features/features.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    breeds = data["breeds"]
    split = data["split"]
    paths = data["paths"]

    # Separar treino/teste
    X_train = X[split == "train"]
    y_train = y[split == "train"]
    X_test = X[split == "test"]
    y_test = y[split == "test"]
    X_test_paths = paths[split == "test"]

    logger.info(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # Padronização
    logger.info("=== PADRONIZANDO FEATURES ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "outputs/models/scaler.pkl")
    logger.info("Scaler salvo em outputs/models/scaler.pkl")

    # Seleção de Features (SelectKBest com f_classif)
    logger.info("=== SELECIONANDO MELHORES FEATURES ===")
    k_best = 50  # Seleciona as 50 melhores features
    selector = SelectKBest(f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    joblib.dump(selector, "outputs/models/selector.pkl")
    logger.info(f"Features reduzidas de {X_train_scaled.shape[1]} para {X_train_selected.shape[1]}")

    # KNN
    logger.info("=== TREINANDO KNN ===")
    k_values = [1, 3, 5, 7, 9, 11, 15]
    knn_cv_scores = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        scores = cross_val_score(
            knn, X_train_selected, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        knn_cv_scores[k] = {
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        }
        logger.info(f"[KNN] k={k} | acc={scores.mean():.4f} ± {scores.std():.4f}")

    # Plot KNN accuracy vs K
    _plot_knn_k_vs_accuracy(knn_cv_scores)

    best_k = max(knn_cv_scores, key=lambda k: knn_cv_scores[k]["mean_accuracy"])
    knn_best = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    knn_best.fit(X_train_selected, y_train)
    joblib.dump(knn_best, "outputs/models/knn_best.pkl")
    logger.info(f"[KNN] Melhor K={best_k} | Modelo salvo")

    # SVM
    logger.info("=== TREINANDO SVM COM GRIDSEARCH ===")
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100, 1000],  # Mais valores de C
        "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1],  # Mais valores de gamma
        "kernel": ["rbf"]
    }

    svm = SVC(random_state=SEED, probability=True)
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train_selected, y_train)

    svm_best = grid_search.best_estimator_
    logger.info(f"[SVM] Melhores hiperparâmetros: {grid_search.best_params_}")
    logger.info(f"[SVM] Melhor acc (CV): {grid_search.best_score_:.4f}")
    joblib.dump(svm_best, "outputs/models/svm_best.pkl")
    logger.info("[SVM] Modelo salvo")

    # Salvar resultados grid search
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv("outputs/results/svm_grid_search.csv", index=False)
    _plot_svm_heatmap(cv_results_df)

    # Avaliação final
    logger.info("=== AVALIAÇÃO NO CONJUNTO DE TESTE ===")
    results_knn = evaluate_model(
        knn_best, X_test_selected, y_test, "KNN", breeds
    )
    results_svm = evaluate_model(
        svm_best, X_test_selected, y_test, "SVM", breeds
    )

    # Comparação final
    comparison = pd.DataFrame({
        "Modelo": ["KNN (baseline)", "SVM (principal)"],
        "Acurácia": [results_knn["accuracy"], results_svm["accuracy"]],
        "F1 Macro": [results_knn["f1_macro"], results_svm["f1_macro"]],
        "F1 Weighted": [results_knn["f1_weighted"], results_svm["f1_weighted"]],
        "Hiperparâm.": [f"k={best_k}", str(grid_search.best_params_)]
    })
    comparison.to_csv("outputs/results/model_comparison.csv", index=False)
    logger.info("\n" + comparison.to_string())

    _plot_model_comparison(comparison)

    # Análise de erros
    y_pred_svm = svm_best.predict(X_test_selected)
    errors_idx = np.where(y_pred_svm != y_test)[0]
    error_df = pd.DataFrame({
        "image_path": X_test_paths[errors_idx],
        "true_breed": [breeds[y_test[i]] for i in errors_idx],
        "predicted": [breeds[y_pred_svm[i]] for i in errors_idx]
    })
    error_df.to_csv("outputs/results/prediction_errors.csv", index=False)
    _plot_error_analysis(error_df, breeds, X_test_paths[errors_idx], y_test[errors_idx], y_pred_svm[errors_idx])

    logger.info("=== CLASSIFICAÇÃO COMPLETA ===")


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    breed_names: np.ndarray
) -> dict:
    """
    Avalia modelo no conjunto de teste, salva métricas e visualizações.

    Parâmetros
    ----------
    model : Classificador treinado
    X_test : np.ndarray
        Features do teste, shape (N, 99)
    y_test : np.ndarray
        Rótulos do teste, shape (N,)
    model_name : str
        Nome do modelo para logging e arquivos
    breed_names : np.ndarray
        Nomes das raças em ordem dos rótulos 0-11

    Retorna
    -------
    dict
        Dicionário com acurácia, f1 macro e f1 weighted
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wei = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"[{model_name}] Accuracy  = {acc:.4f}")
    logger.info(f"[{model_name}] F1 Macro  = {f1_mac:.4f}")
    logger.info(f"[{model_name}] F1 Weight = {f1_wei:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=breed_names, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"outputs/results/{model_name}_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=breed_names, yticklabels=breed_names, ax=ax
    )
    ax.set_title(f"Matriz de Confusão — {model_name}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"outputs/visuals/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    return {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wei}


def _plot_knn_k_vs_accuracy(knn_cv_scores: dict):
    """
    Gera gráfico de acurácia CV vs valores de K para KNN.

    Parâmetros
    ----------
    knn_cv_scores : dict
        Dicionário com scores por K
    """
    k_list = list(knn_cv_scores.keys())
    mean_acc = [knn_cv_scores[k]["mean_accuracy"] for k in k_list]
    std_acc = [knn_cv_scores[k]["std_accuracy"] for k in k_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_list, mean_acc, yerr=std_acc, fmt='-o', capsize=5)
    ax.set_xlabel("Número de Vizinhos (K)")
    ax.set_ylabel("Acurácia (Validação Cruzada 5-fold)")
    ax.set_title("Desempenho do KNN em função de K")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/visuals/knn_k_vs_accuracy.png", dpi=150)
    plt.close()


def _plot_svm_heatmap(cv_results_df: pd.DataFrame):
    """
    Gera heatmap de acurácia CV para C vs gamma do SVM.

    Parâmetros
    ----------
    cv_results_df : pd.DataFrame
        Resultados do GridSearchCV
    """
    # Criar tabela pivot
    cv_results_df["gamma_str"] = cv_results_df["param_gamma"].astype(str)
    pivot = cv_results_df.pivot_table(
        index="param_C", columns="gamma_str", values="mean_test_score"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", ax=ax)
    ax.set_title("Acurácia (CV 5-fold) — SVM RBF")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    plt.tight_layout()
    plt.savefig("outputs/visuals/svm_heatmap_C_gamma.png", dpi=150)
    plt.close()


def _plot_model_comparison(comparison: pd.DataFrame):
    """
    Gera gráfico de barras comparando KNN e SVM.

    Parâmetros
    ----------
    comparison : pd.DataFrame
        Tabela comparativa entre modelos
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.25

    ax.bar(x - width, comparison["Acurácia"], width, label="Acurácia")
    ax.bar(x, comparison["F1 Macro"], width, label="F1 Macro")
    ax.bar(x + width, comparison["F1 Weighted"], width, label="F1 Weighted")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison["Modelo"])
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Desempenho — KNN vs SVM")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("outputs/visuals/model_comparison_bar.png", dpi=150)
    plt.close()


def _plot_error_analysis(error_df: pd.DataFrame, breeds: np.ndarray, error_paths: np.ndarray, true_y: np.ndarray, pred_y: np.ndarray):
    """
    Gera grade visual com 12 erros representativos (1 por raça, se possível).

    Parâmetros
    ----------
    error_df : pd.DataFrame
        DataFrame com erros de predição
    breeds : np.ndarray
        Nomes das raças
    error_paths : np.ndarray
        Caminhos das imagens com erro
    true_y : np.ndarray
        Raças reais
    pred_y : np.ndarray
        Raças preditas
    """
    # Selecionar 1 erro por raça, se disponível
    selected = []
    for breed in breeds:
        breed_errors = error_df[error_df["true_breed"] == breed]
        if len(breed_errors) > 0:
            selected.append(breed_errors.iloc[0])

    if len(selected) == 0:
        logger.warning("Nenhum erro para plotar análise de erros")
        return

    # Plotar
    n = min(len(selected), 12)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i in range(n):
        row = selected[i]
        img_bgr = cv2.imread(row["image_path"])
        if img_bgr is None:
            axes[i].set_title("Imagem não carregada")
            axes[i].axis("off")
            continue
        img_rgb = img_bgr[:, :, ::-1]
        img_rgb = resize_rgb(img_rgb, (128, 128))
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Real: {row['true_breed']}\nPredito: {row['predicted']}", fontsize=10)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/visuals/error_analysis.png", dpi=150)
    plt.close()


# Import cv2 para o _plot_error_analysis (apenas para leitura)
import cv2

if __name__ == "__main__":
    run_classification_pipeline()