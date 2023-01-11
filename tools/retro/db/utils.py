# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import glob
import h5py
import json
import numpy as np
import os
from tqdm import tqdm

from megatron import get_retro_args, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

from .dataset import DBDataset


def get_base_db_workdir():
    '''Sub-directory for DB data.'''
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "db")


def get_indexed_dataset_infos_path():
    '''Path to indexed dataset meta-infos.'''
    return os.path.join(get_base_db_workdir(), "indexed_dataset_infos.json")


def save_indexed_dataset_infos(indexed_dataset_infos):
    '''Save dataset order & meta-info.'''

    # Remove 'dataset' field.
    clean_infos = []
    for info in indexed_dataset_infos:
        info = dict(info)
        del info["dataset"]
        clean_infos.append(info)

    # Save.
    with open(get_indexed_dataset_infos_path(), "w") as f:
        json.dump(clean_infos, f, indent = 4)


def get_indexed_dataset_infos():
    '''Load indexed dataset meta-infos.'''

    # Load json.
    path = get_indexed_dataset_infos_path()
    with open(path) as f:
        infos = json.load(f)

    # Add indexed datasets.
    for info in infos:
        info["dataset"] = make_indexed_dataset(info["prefix"], "mmap", True)

    return infos


def get_individual_db_dir(name):
    '''Individual DB's directory.'''
    return os.path.join(get_base_db_workdir(), "individual", name, "db")


def get_individual_db(ds_id, ds_info):
    '''Load individual dataset's chunk DB.'''
    db_paths = sorted(glob.glob(ds_info["db_dir"] + "/*hdf5"))
    # *Note*: convert to dataset, rather than copying to memory.
    db = np.zeros((ds_info["n_chunks"], 5), dtype = "i8")
    db[:, 0] = ds_id
    start_idx = 0
    for db_path in db_paths:
        f = h5py.File(db_path, "r")
        n_chunks_current = f["chunks_valid"].shape[0]
        db[start_idx:(start_idx+n_chunks_current), 1:] = f["chunks_valid"]
        start_idx += n_chunks_current
        f.close()

    assert start_idx == ds_info["n_chunks"]

    return db


def get_merged_db_path_map():
    '''Paths to merged datasets.'''
    base_dir = get_base_db_workdir()
    return {
        "sampled" : os.path.join(base_dir, "merged", "sampled.hdf5"),
        "train" : os.path.join(base_dir, "merged", "train.hdf5"),
        "valid" : os.path.join(base_dir, "merged", "valid.hdf5"),
    }


def get_merged_dataset(db_type, indexed_dataset_infos = None):
    '''Get merged dataset.'''

    args = get_retro_args()

    if not indexed_dataset_infos:
        indexed_dataset_infos = get_indexed_dataset_infos()

    # Load chunks.
    db_path = get_merged_db_path_map()[db_type]
    f = h5py.File(db_path, "r")
    chunks = f["chunks"]

    # DB dataset.
    indexed_datasets = [ info["dataset"] for info in indexed_dataset_infos ]
    dataset = DBDataset(db_path, indexed_datasets, chunks,
                        args.retro_gpt_chunk_length)

    return dataset


def get_merged_sampled_dataset(indexed_dataset_infos = None):
    return get_merged_dataset("sampled", indexed_dataset_infos)


def get_merged_train_dataset(indexed_dataset_infos = None):
    return get_merged_dataset("train", indexed_dataset_infos)


def get_merged_valid_dataset(indexed_dataset_infos = None):
    return get_merged_dataset("valid", indexed_dataset_infos)


def get_train_doc_chunk_map_dir():
    dirname = os.path.join(get_base_db_workdir(), "merged", "train_doc_chunk_map")
    os.makedirs(dirname, exist_ok = True)
    return dirname


def get_train_doc_chunk_map():

    paths = sorted(glob.glob(get_train_doc_chunk_map_dir() + "/*.json"))

    doc_map = defaultdict(set)
    for path in tqdm(paths, "load train doc maps"):

        # Read file.
        # crnt_doc_map = json.loads(zlib.decompress(data).decode())
        with open(path) as f:
            crnt_doc_map = json.load(f)

        # Add to doc map.
        for key, chunk_ids in crnt_doc_map.items():
            key = tuple(int(i) for i in key.split(","))
            doc_map[key].update(chunk_ids)

    return doc_map
