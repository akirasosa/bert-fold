import concurrent
import re
from concurrent.futures.process import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from const import DATA_PROTEIN_NET_DIR
from mylib.torch.functional import calculate_torsions

MAX_SEQUENCE_LENGTH = 2000


def _calculate_psi(item):
    R = _concat_coords(item)
    R = torch.from_numpy(R)
    size = len(R[3::3])
    torsions = calculate_torsions(
        R[0::3][:size],
        R[1::3][:size],
        R[2::3][:size],
        R[3::3][:size],
    )

    return torsions.numpy()


def _calculate_phi(item):
    R = _concat_coords(item)
    R = torch.from_numpy(R)
    size = len(R[5::3])
    torsions = calculate_torsions(
        R[2::3][:size],
        R[3::3][:size],
        R[4::3][:size],
        R[5::3][:size],
    )

    return torsions.numpy()


def _concat_coords(item):
    R_n = item['tertiary_n'].reshape(-1, 3)
    R_ca = item['tertiary_ca'].reshape(-1, 3)
    R_cb = item['tertiary_cb'].reshape(-1, 3)
    R = np.stack((R_n, R_ca, R_cb), axis=-1).transpose((0, 2, 1)).reshape(-1, 3)
    return R


def read_protein_from_file(file_pointer, handle_missing=True):
    """The algorithm Defining Secondary Structure of Proteins (DSSP) uses information on e.g. the
    position of atoms and the hydrogen bonds of the molecule to determine the secondary structure
    (helices, sheets...).
    """
    dict_ = {}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            # primary = encode_primary_string(file_pointer.readline()[:-1])
            primary = file_pointer.readline()[:-1]
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for _residue in range(21):
                evolutionary.append([float(step) for step in file_pointer.readline().split()])
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            # secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
            secondary = file_pointer.readline()[:-1]
            dict_.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimension
            for _axis in range(3):
                tertiary.append([float(coord) for coord in file_pointer.readline().split()])
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
            mask_str = ''.join(map(str, mask))

            if handle_missing:
                print("-------------")
                sequence_end = len(mask)  # for now, assume no C-terminal truncation needed
                print("Reading the protein " + id_)
                if re.search(r'1+0+1+', mask_str) is not None:  # indicates missing coordinates
                    print("One or more internal coordinates missing. Protein is discarded.")
                elif re.search(r'^0*$', mask_str) is not None:  # indicates no coordinates at all
                    print("One or more internal coordinates missing. It will be discarded.")
                else:
                    if mask[0] == 0:
                        print("Missing coordinates in the N-terminal end. Truncating protein.")
                    # investigate when the sequence with coordinates start and finish
                    sequence_start = re.search(r'1', mask_str).start()
                    if re.search(r'10', mask_str) is not None:  # missing coords in the C-term end
                        sequence_end = re.search(r'10', mask_str).start() + 1
                        print("Missing coordinates in the C-term end. Truncating protein.")
                    print("Analyzing amino acids", sequence_start + 1, "-", sequence_end)

                    # split lists in dict to have the seq with coords
                    # separated from what should not be analysed
                    if 'secondary' in dict_:
                        dict_.update({'secondary': secondary[sequence_start:sequence_end]})
                    dict_.update({'primary': primary[sequence_start:sequence_end]})
                    dict_.update({'mask': mask[sequence_start:sequence_end]})
                    for elem in range(len(dict_['evolutionary'])):
                        dict_['evolutionary'][elem] = \
                            dict_['evolutionary'][elem][sequence_start:sequence_end]
                    for elem in range(len(dict_['tertiary'])):
                        dict_['tertiary'][elem] = \
                            dict_['tertiary'][elem][sequence_start * 3:sequence_end * 3]
        elif next_line == '\n':
            return dict_
        elif next_line == '':
            if dict_:
                return dict_
            else:
                return None


def process_file(input_file: Path, handle_missing=False):
    print("Processing raw data file", input_file)

    results = []

    with input_file.open() as fp:
        while True:
            next_protein = read_protein_from_file(fp, handle_missing)

            if next_protein is None:
                break

            sequence_length = len(next_protein['primary'])

            if sequence_length > MAX_SEQUENCE_LENGTH:
                print("Dropping protein as length too long:", sequence_length)
                continue

            tertiary = np.array(next_protein['tertiary']).reshape((3, -1, 3))
            tertiary /= 100.  # Now, this is angstrom.

            result = {
                'secondary': None,
                **next_protein,
                'tertiary_n': tertiary[:, :, 0].T.reshape(-1),
                'tertiary_ca': tertiary[:, :, 1].T.reshape(-1),
                'tertiary_cb': tertiary[:, :, 2].T.reshape(-1),
                'valid_mask': np.array(next_protein['mask']).reshape(-1),
                'evolutionary': np.array(next_protein['evolutionary']).reshape(-1),
            }
            del result['tertiary']
            del result['mask']
            results.append(result)

    df = pd.DataFrame(results)
    df['phi'] = df.apply(_calculate_phi, axis=1)
    df['psi'] = df.apply(_calculate_psi, axis=1)

    df.to_parquet(f'{input_file}.pqt')


# %%
if __name__ == '__main__':
    # %%
    base_path = Path('/mnt/nfs/vfa-red/akirasosa/data/ProteinNet/casp12')
    files = list(filter(lambda x: not str(x).endswith('pqt'), base_path.glob('*')))
    files = sorted(files)

    # %%
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    jobs = [executor.submit(process_file, f) for f in files]
    for job in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs)):
        job.result()

    # %%
    df_train = pd.concat([
        pd.read_parquet(f)
        for f in base_path.glob('training*.pqt')
    ], ignore_index=True)

    dup = df_train[['id', 'primary']].duplicated()
    df_train = df_train[~dup]

    df_train.to_parquet(base_path / 'train_all.pqt')
