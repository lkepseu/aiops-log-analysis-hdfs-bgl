# configs/parsing_config.py
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DrainConfig:
    """
    Configuration de parsing pour un dataset donné.
    """
    dataset_name: str           # ex: "HDFS" ou "BGL"
    log_file: str               # nom du fichier brut (dans data/raw/)
    log_format: str             # format de log pour Drain
    indir: str                  # dossier d'entrée (logs bruts)
    outdir: str                 # dossier de sortie (CSV structurés)
    depth: int                  # profondeur de l'arbre Drain
    st: float                   # seuil de similarité
    rex: List[str] = field(default_factory=list)  # liste de regex pour variables fréquemment rencontrées


def get_parsing_configs(base_input_dir: str = "data/raw",
                        base_output_dir: str = "data/parsed") -> Dict[str, DrainConfig]:
    """
    Retourne les configurations pour les datasets supportés.
    Les clés du dict sont les noms des datasets ("HDFS", "BGL").
    """

    # 1) HDFS
    # Format :
    # <Date> <Time> <Pid> <Level> <Component>: <Content>
    #
    # Exemple logique :
    # 2009-07-20 11:54:51,821 12345 INFO org.apache.hadoop.hdfs.server.namenode.FSNamesystem MESSAGE...
    hdfs_config = DrainConfig(
        dataset_name="HDFS",
        log_file="HDFS.log",
        log_format="<Date> <Time> <Pid> <Level> <Component>: <Content>",
        indir=base_input_dir,
        outdir=f"{base_output_dir}/HDFS",
        depth=4,
        st=0.5,
        rex=[
            # 1) IP avant le port (ex: 10.250.10.6:50010)
            #    -> on remplace seulement l'IP, pas le ':'
            #    10.250.10.6:50010  => <*>:50010
            r"(\d{1,3}(?:\.\d{1,3}){3})(?=:)",

            # 2) IP précédée d'un slash (ex: /10.250.18.114)
            #    -> /10.250.18.114  => /<*>
            r"(?<=/)(\d{1,3}(?:\.\d{1,3}){3})",

            # 3) Port après un ':' (ex: :50010)
            #    -> :50010  => :<*>
            r"(?<=:)\d+",

            # 4) Numéro de bloc HDFS, en gardant le préfixe 'blk_'
            #    blk_-5974833545991408899  => blk_<*>
            r"(?<=blk_)[\-]?\d+",

            # 5) ID de tâche MapReduce, en gardant '_task_'
            #    _task_200811092030_0001_m_000590_0  => _task_<*>
            r"(?<=_task_)\d{12}_\d{4}_[mr]_\d{6}_\d",

            # 6) Taille après 'size ' en gardant le mot 'size'
            #    size 67108864  => size <*>
            r"(?<=size\s)\d+"
        ],
    )


    # 2) BGL (Blue Gene/L)
    # Format :
    # <Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>
    #
    # Exemple logique :
    # KERN 2005-08-22-11.50.11.486431 2005-08-22 R34-M1-N8-I 11:50:11.486431 1 RAS KERNEL INFO MESSAGE...
    bgl_config = DrainConfig(
        dataset_name="BGL",
        log_file="BGL.log",
        log_format="<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        indir=base_input_dir,
        outdir=f"{base_output_dir}/BGL",
        depth=4,
        st=0.3,
        rex=[
                # 1) Timestamp complet BGL : 2005-08-22-11.50.11.486431
                r"\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6}",

                # 2) Date type 2005.08.22
                r"\d{4}\.\d{2}\.\d{2}",

                # 3) Identifiant de noeud Blue Gene/L : R34-M1-N8-I:J18-U01 etc.
                r"R\d+-M\d+-N[\w-]+(:J\d+-U\d+)?",

                # 4) Registres flottants : fpr0, fpr1, ..., fpr31  → remplacés par <*>
                #    (ça va fusionner une grosse quantité de templates ultra rares)
                r"fpr\d+",

                # 5) Gros mots hexadécimaux nus (adresses, flottants encodés, etc.)
                #    ex : ffffffff, 3ff00000, b1145000
                r"\b[0-9A-Fa-f]{8,16}\b",

                # 6) Valeurs hexadécimales avec préfixe 0x
                r"0x[0-9A-Fa-f]+",

                # 7) Entiers décimaux isolés (compteurs, NodeRepeat, offsets, etc.)
                r"\b\d+\b",
            ],
    )


    return {
        "HDFS": hdfs_config,
        "BGL": bgl_config,
    }
