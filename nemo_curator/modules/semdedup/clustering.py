# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pathlib
import pprint
from datetime import datetime
from typing import Tuple

import cupy as cp
import dask_cudf
import numpy as np
from cuml.dask.cluster import KMeans
from dask.distributed import wait

from nemo_curator.modules.semdedup.utils import get_logger, parse_arguments
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.script_utils import parse_client_args


def get_embedding_ar(df: "cudf.DataFrame") -> cp.ndarray:
    return df["embeddings"].list.leaves.values.reshape(len(df), -1)


def add_dist_to_cents(df: "cudf.DataFrame", centriods: cp.ndarray) -> "cudf.DataFrame":
    embed_array = get_embedding_ar(df)
    centroids_ar = centriods[df["nearest_cent"].values]
    dist_to_cents = cp.sqrt(np.sum((embed_array - centroids_ar) ** 2, axis=1))
    df["dist_to_cent"] = dist_to_cents
    return df


def compute_centroids(
    args: argparse.Namespace, logger: logging.Logger, client: "dask.distributed.Client"
) -> Tuple[cp.ndarray, "dask_cudf.DataFrame"]:

    # Kmeans can only be done with L2 using cuML.
    assert args.clustering["Kmeans_with_cos_dist"] == False

    ## -- Load clustering parameters
    root_dir = args.root
    emb_pqt_loc = os.path.join(args.root, args.embeddings["save_loc"])
    emb_size = args.embeddings["emb_size"]
    niter = args.clustering["niter"]
    ncentroids = args.clustering["num_clusters"]
    num_workers = get_num_workers(client)

    save_folder = os.path.join(root_dir, args.clustering["save_loc"])
    os.makedirs(save_folder, exist_ok=True)

    with open(pathlib.Path(save_folder, "clustering_params.txt"), "w") as fout:
        pprint.pprint(args, fout)

    ddf = dask_cudf.read_parquet(
        emb_pqt_loc,
        columns=["embeddings", args.id_col["name"]],
        split_row_groups=False,
    )
    logger.info(f"Total number of partitions in input_ddf: {ddf.npartitions}")
    # Persist ddf to save IO costs in host memory, should be able to do this via
    # spilling to (TODO)
    ddf = ddf.to_backend("pandas").persist()
    ddf = ddf.repartition(npartitions=num_workers * 4)
    wait(ddf)
    client.rebalance(ddf)
    # Switch back to GPU
    ddf = ddf.to_backend("cudf")

    cupy_darr = ddf.map_partitions(get_embedding_ar, meta=cp.ndarray([1, emb_size]))

    cupy_darr.compute_chunk_sizes()

    kmeans = KMeans(n_clusters=ncentroids, max_iter=niter, oversampling_factor=10)
    logger.info("KMeans starting fit")
    kmeans.fit(cupy_darr)
    logger.info("KMeans fit complete")

    logger.info("Computing nearest centroids using kmeans.predict")
    nearest_cents = kmeans.predict(cupy_darr)
    ddf["nearest_cent"] = nearest_cents.astype(np.int32)
    logger.info("Nearest centroids computed")

    logger.info("Computing distances to centroids using add_dist_to_cents")
    meta_df = ddf._meta.copy()
    meta_df["dist_to_cent"] = cp.zeros(1)
    ddf = ddf.map_partitions(
        add_dist_to_cents, centriods=kmeans.cluster_centers_, meta=meta_df
    )
    logger.info("Distances to centroids computed")

    centroids = kmeans.cluster_centers_
    logger.info("Centroids computed")

    ddf = ddf.reset_index(drop=True)

    logger.info("Saving centroids complete")
    kmeans_centroids_file = pathlib.Path(save_folder, "kmeans_centroids.npy")
    np.save(kmeans_centroids_file, centroids)
    return ddf


if __name__ == "__main__":
    # Configure command line arguments
    args = parse_arguments()

    # TODO: Cleanup below
    save_folder = f'{args.root}/{args.clustering["save_loc"]}'
    os.makedirs(save_folder, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(save_folder, "compute_centroids.log")
    logger = get_logger(
        file_name=log_file,
        level=logging.INFO,
        stdout=True,
    )

    client = get_client(**parse_client_args(args))
    dt1 = datetime.now()
    print("Start time:", dt1)
    ddf = compute_centroids(args, logger, client)
    ddf.to_parquet(
        f"{save_folder}/embs_by_nearest_center/",
        index=False,
        partition_on="nearest_cent",
    )
    del ddf
    dt2 = datetime.now()
    elapse = dt2 - dt1
    print("End time:", dt2)
    print("elapse:", elapse)

    client.cancel(client.futures, force=True)
    client.close()
