import pandas as pd

def remap_event_ids(templates_path: str, structured_path: str) -> None:
    """
    Remappe les EventId hexadécimaux produits par Drain en identifiants
    symboliques lisibles : E1, E2, E3, ...

    - Trie les templates par Occurrences (desc)
    - Crée un mapping EventId_hex -> E1..En
    - Applique le mapping au fichier structuré et au fichier templates
    - Réécrit les CSV en place
    """

    # Charger les fichiers
    df_templates = pd.read_csv(templates_path)
    df_struct    = pd.read_csv(structured_path)

    # Vérifications minimales
    if "EventId" not in df_templates.columns:
        raise ValueError("templates.csv : colonne 'EventId' absente.")
    if "Occurrences" not in df_templates.columns:
        raise ValueError("templates.csv : colonne 'Occurrences' absente.")
    if "EventId" not in df_struct.columns:
        raise ValueError("structured.csv : colonne 'EventId' absente.")

    # Tri par Occurrences descendantes et re-indexation
    df_templates = df_templates.sort_values("Occurrences", ascending=False).reset_index(drop=True)

    # Génération de E1, E2, E3, ...
    df_templates["NewEventId"] = ["E" + str(i + 1) for i in range(len(df_templates))]

    # Mapping old -> new
    mapping = dict(zip(df_templates["EventId"], df_templates["NewEventId"]))

    # Remplacement dans le structured.csv
    df_struct["EventId"] = df_struct["EventId"].map(mapping)

    # Remplacement dans templates.csv
    df_templates["EventId"] = df_templates["NewEventId"]
    df_templates.drop(columns=["NewEventId"], inplace=True)

    # Sauvegarde
    df_templates.to_csv(templates_path, index=False)
    df_struct.to_csv(structured_path, index=False)