from pathlib import Path

DATA_DIR = Path('../data')
DATA_PROTEIN_NET_DIR = DATA_DIR / 'ProteinNet'

TMP_DIR = Path('../tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = Path('../experiments')
