from configs.analysis_failure_hdfs import plot_eventid_histogram_corrupted
from configs.analysis_failure_hdfs import plot_eventid_histogram_normal
from configs.analysis_failure_hdfs import plot_eventid_normal_vs_corrupted
from configs.analysis_failure_hdfs import load_and_split_hdfs_matrix
from configs.analysis_failure_hdfs import plot_blockid_normal_vs_corrupted_ratio
from configs.analysis_failure_bgl import identify_failure_event_ids
from configs.analysis_failure_bgl import plot_window_anomaly_count

import argparse


def analyze_failures_repartition_hdfs(matrix_csv, labels_csv, top_k=20):
    df_normaux, df_corrompus, event_cols = load_and_split_hdfs_matrix(
        matrix_csv, labels_csv
    )

    plot_eventid_histogram_corrupted(df_corrompus, event_cols, top_k)
    plot_eventid_histogram_normal(df_normaux, event_cols, top_k)
    plot_eventid_normal_vs_corrupted(df_normaux, df_corrompus, event_cols, top_k)
    plot_blockid_normal_vs_corrupted_ratio(labels_csv)

def analyze_failures_repartition_bgl(
    matrix_csv: str,
    structured_csv: str,
) -> None:
    # 1) Identifier les EventId de failure depuis le structuré
    failure_event_ids = identify_failure_event_ids(structured_csv)

    print(f"[INFO] Nombre d'EventId de failure détectés : {len(failure_event_ids)}")
    if not failure_event_ids:
        print("[WARN] Aucun EventId de failure détecté, arrêt de l'analyse.")
        return

    # 2) Histogramme des fenêtres normales vs anormales
    plot_window_anomaly_count(
        matrix_csv=matrix_csv,
        failure_event_ids=failure_event_ids,
        title="BGL — Fenêtres normales vs anormales (count plot)",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse des anomalies pour HDFS ou BGL")

    parser.add_argument("--dataset", type=str, required=True, choices=["hdfs", "bgl"], help="Nom du dataset : hdfs ou bgl")
    parser.add_argument("--matrix_csv", type=str, required=True, help="Chemin vers le fichier *_matrix.csv")
    parser.add_argument("--labels_csv", type=str, required=False, help="Chemin vers le fichier *_label.csv (HDFS seulement)")
    parser.add_argument("--structured_csv", type=str, required=False, help="Chemin vers le fichier *_structured.csv (BGL seulement)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k EventId à afficher")

    args = parser.parse_args()

    print(f"=== Analyse du dataset : {args.dataset.upper()} ===")
    print(f"[INFO] matrix_csv     : {args.matrix_csv}")
    print(f"[INFO] labels_csv     : {args.labels_csv}")
    print(f"[INFO] structured_csv : {args.structured_csv}")


    if args.dataset == "hdfs":
        if not args.labels_csv:
            raise ValueError("Pour HDFS, vous devez fournir --labels_csv")
        print(f"[INFO] top_k :{args.top_k}")
        analyze_failures_repartition_hdfs(
            matrix_csv=args.matrix_csv,
            labels_csv=args.labels_csv,
            top_k=args.top_k,
        )

    elif args.dataset == "bgl":
        if not args.structured_csv:
            raise ValueError("Pour BGL, vous devez fournir --structured_csv")
        analyze_failures_repartition_bgl(
            matrix_csv=args.matrix_csv,
            structured_csv=args.structured_csv,
        )

    print("=== Analyse terminée ===")