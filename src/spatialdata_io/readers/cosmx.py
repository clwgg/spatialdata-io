from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from dask_image.imread import imread
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from skimage.transform import estimate_transform
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations.transformations import Affine, Identity

from spatialdata_io._constants._constants import CosmxKeys
from spatialdata_io._docs import inject_docs

__all__ = ["cosmx"]


def _infer_flip_y(obs):
    """
    infer if images/labels need to be flipped to match points (early data) or not (later data)
    """
    y_coord_corr = (
        obs[['fov', 'CenterY_local_px', 'CenterY_global_px']]
        .groupby('fov', observed=False).corr().reset_index()
        .query('level_1 == "CenterY_local_px"')[['fov', 'CenterY_global_px']]
        .reset_index(drop=True).rename(columns={'CenterY_global_px': 'y_corr'})
    )
    if y_coord_corr.y_corr.mean() < 0:
        flip_y = False
    else:
        flip_y = True
    return flip_y

def _get_csv_name(path):
    path = path.with_suffix(".csv")
    if not path.exists():
        path = path.with_suffix(".csv.gz")
        if not path.exists():
            path = None
    return path

def _get_tx_dtypes():
    return {
        'fov': 'int64',
        'cell_ID': 'int64',
        'cell': 'O',
        'x_local_px': 'float64',
        'y_local_px': 'float64',
        'x_global_px': 'float64',
        'y_global_px': 'float64',
        'z': 'int64',
        'target': 'O',
        'CellComp': 'O',
    }


@inject_docs(cx=CosmxKeys)
def cosmx(
    path: str | Path,
    dataset_id: str | None = None,
    transcripts: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """
    Read *Cosmx Nanostring* data.

    This function reads the following files:

        - ``<dataset_id>_`{cx.COUNTS_SUFFIX!r}```: Counts matrix.
        - ``<dataset_id>_`{cx.METADATA_SUFFIX!r}```: Metadata file.
        - ``<dataset_id>_`{cx.FOV_SUFFIX!r}```: Field of view file.
        - ``{cx.IMAGES_DIR!r}``: Directory containing the images.
        - ``{cx.LABELS_DIR!r}``: Directory containing the labels.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    dataset_id
        Name of the dataset.
    transcripts
        Whether to also read in transcripts information.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    # tries to infer dataset_id from the name of the counts file
    if dataset_id is None:
        counts_files = [f for f in os.listdir(path) if str(f).endswith(CosmxKeys.COUNTS_SUFFIX)]
        if len(counts_files) == 1:
            found = re.match(rf"(.*)_{CosmxKeys.COUNTS_SUFFIX}", counts_files[0])
            if found:
                dataset_id = found.group(1)
    if dataset_id is None:
        raise ValueError("Could not infer `dataset_id` from the name of the counts file. Please specify it manually.")

    # check for file existence
    counts_file = _get_csv_name(path / f"{dataset_id}_{CosmxKeys.COUNTS_SUFFIX}")
    if not counts_file:
        raise FileNotFoundError(f"Counts file not found: {counts_file}.")
    if transcripts:
        transcripts_file = _get_csv_name(path / f"{dataset_id}_{CosmxKeys.TRANSCRIPTS_SUFFIX}")
        if not transcripts_file:
            raise FileNotFoundError(f"Transcripts file not found: {transcripts_file}.")
    else:
        transcripts_file = None
    meta_file = _get_csv_name(path / f"{dataset_id}_{CosmxKeys.METADATA_SUFFIX}")
    if not meta_file:
        raise FileNotFoundError(f"Metadata file not found: {meta_file}.")
    fov_file = _get_csv_name(path / f"{dataset_id}_{CosmxKeys.FOV_SUFFIX}")
    if not fov_file:
        raise FileNotFoundError(f"Found field of view file: {fov_file}.")
    images_dir = path / CosmxKeys.IMAGES_DIR
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}.")
    labels_dir = path / CosmxKeys.LABELS_DIR
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}.")

    # Read obs first (typically smaller than counts)
    obs = pd.read_csv(path / meta_file, header=0, index_col=CosmxKeys.INSTANCE_KEY)
    obs[CosmxKeys.FOV] = pd.Categorical(obs[CosmxKeys.FOV].astype(str))
    obs[CosmxKeys.REGION_KEY] = pd.Categorical(obs[CosmxKeys.FOV].astype(str).apply(lambda s: s + "_labels"))
    obs[CosmxKeys.INSTANCE_KEY] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[CosmxKeys.FOV].values, sep="_")

    flip_y = _infer_flip_y(obs)

    # Drop the `cell_id` column because it throws an error given the presence of the `cell_ID` column.
    # Also: `cell_id` is redundant with `cell`
    obs.drop(columns="cell_id", inplace=True)

    # Read only index and FOV columns to build counts index without loading full data
    # This is memory-efficient for large files
    logger.info("Reading counts file index to determine row mapping...")
    counts_index_df = pd.read_csv(
        path / counts_file,
        header=0,
        usecols=[CosmxKeys.INSTANCE_KEY, CosmxKeys.FOV],
        dtype={CosmxKeys.INSTANCE_KEY: str, CosmxKeys.FOV: str},
    )
    counts_index = (
        counts_index_df[CosmxKeys.INSTANCE_KEY].astype(str)
        .str.cat(counts_index_df[CosmxKeys.FOV].astype(str).values, sep="_")
    )
    del counts_index_df  # Free memory

    # Get column names and compute common_index
    counts_header = pd.read_csv(path / counts_file, header=0, nrows=0)
    counts_columns = [col for col in counts_header.columns if col not in [CosmxKeys.INSTANCE_KEY, CosmxKeys.FOV]]
    common_index = obs.index.intersection(counts_index)
    common_index_set = set(common_index)  # For faster lookup
    del counts_header  # Free memory

    # Read counts data in chunks and convert to sparse immediately
    # This avoids loading the entire dense DataFrame into memory
    logger.info(f"Reading counts data in chunks (found {len(common_index)} common cells)...")
    chunk_size = 50000  # Process 50k rows at a time
    
    # Create a mapping from modified index to position in common_index for proper ordering
    common_index_map = {idx: i for i, idx in enumerate(common_index)}
    
    # Store sparse data with their target positions
    sparse_data_list = []  # List of (sparse_matrix, position_array)
    
    # Read in chunks using pandas chunked reading
    counts_reader = pd.read_csv(
        path / counts_file,
        header=0,
        index_col=CosmxKeys.INSTANCE_KEY,
        chunksize=chunk_size,
    )
    
    for chunk_df in counts_reader:
        # Modify index same way as before
        chunk_df.index = chunk_df.index.astype(str).str.cat(
            chunk_df.pop(CosmxKeys.FOV).astype(str).values, sep="_"
        )
        # Filter to common_index
        chunk_mask = chunk_df.index.isin(common_index_set)
        if chunk_mask.any():
            chunk_filtered = chunk_df.loc[chunk_mask, counts_columns]
            # Convert to sparse immediately to save memory
            sparse_chunk = csr_matrix(chunk_filtered.values)
            # Get positions in common_index for this chunk
            chunk_indices = chunk_filtered.index
            chunk_positions = np.array([common_index_map[idx] for idx in chunk_indices])
            sparse_data_list.append((sparse_chunk, chunk_positions))
        del chunk_df  # Free memory

    # Build final sparse matrix in correct order
    if sparse_data_list:
        # Combine all sparse matrices
        all_sparse = sparse_vstack([data[0] for data in sparse_data_list], format="csr")
        all_positions = np.concatenate([data[1] for data in sparse_data_list])
        
        # Reorder to match common_index order
        reorder_idx = np.argsort(all_positions)
        counts_sparse = all_sparse[reorder_idx]
    else:
        # Edge case: no matching rows
        counts_sparse = csr_matrix((len(common_index), len(counts_columns)))

    adata = AnnData(
        counts_sparse,
        obs=obs.loc[common_index, :],
    )
    adata.var_names = counts_columns

    # Filter out one-cell FOVs since we cannot define a transform to global from a single cell
    num_cells = adata.obs[['fov']].groupby('fov', observed=False).size()
    adata = adata[adata.obs['fov'].isin(num_cells[num_cells > 2].index)].copy()

    table = TableModel.parse(
        adata,
        region=list(set(adata.obs[CosmxKeys.REGION_KEY].astype(str).tolist())),
        region_key=CosmxKeys.REGION_KEY.value,
        instance_key=CosmxKeys.INSTANCE_KEY.value,
    )

    fovs_counts = list(map(str, adata.obs.fov.astype(int).unique()))

    affine_transforms_to_global = {}

    for fov in fovs_counts:
        idx = table.obs.fov.astype(str) == fov
        loc = table[idx, :].obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].values
        glob = table[idx, :].obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].values
        out = estimate_transform(ttype="affine", src=loc, dst=glob)
        affine_transforms_to_global[fov] = Affine(
            # out.params, input_coordinate_system=input_cs, output_coordinate_system=output_cs
            out.params,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

    table.obsm["global"] = table.obs[[CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL]].to_numpy()
    table.obsm["spatial"] = table.obs[[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL]].to_numpy()
    table.obs.drop(
        columns=[CosmxKeys.X_LOCAL_CELL, CosmxKeys.Y_LOCAL_CELL, CosmxKeys.X_GLOBAL_CELL, CosmxKeys.Y_GLOBAL_CELL],
        inplace=True,
    )

    # prepare to read images and labels
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")
    file_extensions += tuple(ext.upper() for ext in file_extensions)
    pat = re.compile(r".*_F(\d+)")

    # check if fovs are correct for images and labels
    fovs_images = []
    for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fovs_images.append(str(int(pat.findall(fname)[0])))

    fovs_labels = []
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fovs_labels.append(str(int(pat.findall(fname)[0])))

    fovs_images_and_labels = set(fovs_images).intersection(set(fovs_labels))
    fovs_diff = fovs_images_and_labels.difference(set(fovs_counts))
    if len(fovs_diff):
        logger.warning(
            f"Found images and labels for {len(fovs_images)} FOVs, but only {len(fovs_counts)} FOVs in the counts file.\n"
            + f"The following FOVs are missing: {fovs_diff} \n"
            + "... will use only fovs in Table."
        )

    logger.info("Reading images...")

    channels = [c.replace("Max.", "") for c in
                table.obs.columns[table.obs.columns.str.startswith("Max.")]]
    channels = [re.sub("^Membrane.*$", "Membrane", c) for c in channels]
    original_channels = channels.copy()
    if len(channels) < 5:
        raise ValueError(f"Need names for at least 5 channels. Found only {len(channels)}: {channels}")
    elif len(channels) > 5:
        logger.warning(
            f"(TRUNCATED) More than 5 channel names detected; truncating channel names from {original_channels} "
            f"to {channels[:5]}."
        )
        channels = channels[:5]
    else:
        logger.info(
            f"Found exactly 5 channel names: {channels}"
        )

    # read images
    images = {}
    for fname in os.listdir(path / CosmxKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                im = imread(path / CosmxKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze()
                if flip_y:
                    matched_im = da.flip(im, axis=1)
                else:
                    matched_im = im
                parsed_im = Image2DModel.parse(
                    matched_im,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_image": aff,
                    },
                    dims=("c", "y", "x"),
                    c_coords=channels,
                    rgb=None,
                    **image_models_kwargs,
                )
                images[f"{fov}_image"] = parsed_im
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")

    # read labels
    logger.info("Reading labels...")
    labels = {}
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                la = imread(path / CosmxKeys.LABELS_DIR / fname, **imread_kwargs).squeeze()
                if flip_y:
                    matched_la = da.flip(la, axis=0)
                else:
                    matched_la = la
                parsed_la = Labels2DModel.parse(
                    matched_la,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_labels": aff,
                    },
                    dims=("y", "x"),
                    **image_models_kwargs,
                )
                labels[f"{fov}_labels"] = parsed_la
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping labels {fname}.")

    points: dict[str, DaskDataFrame] = {}
    if transcripts:
        # convert the .csv to .parquet and read it with pyarrow.parquet for faster subsetting
        # Use pandas chunked reading to handle gzipped files efficiently
        import tempfile
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Converting transcripts .csv to .parquet to improve the speed of the slicing operations...")
            assert transcripts_file is not None
            
            # Read transcripts file in chunks using pandas (handles gzip efficiently)
            # Filter by FOV and write to parquet incrementally to avoid loading all data into memory
            parquet_path = Path(tmpdir) / "transcripts.parquet"
            chunk_size = 100000  # Process 100k rows at a time
            fovs_counts_int = set(int(fov) for fov in fovs_counts)  # Convert to int for filtering
            
            transcripts_reader = pd.read_csv(
                path / transcripts_file,
                header=0,
                dtype=_get_tx_dtypes(),
                chunksize=chunk_size,
            )
            
            # Write filtered chunks directly to parquet incrementally
            # This avoids loading all filtered data into memory at once
            parquet_writer = None
            first_chunk = True
            total_rows_written = 0
            
            for chunk_df in transcripts_reader:
                # Filter to only FOVs we care about
                chunk_filtered = chunk_df[chunk_df[CosmxKeys.FOV].isin(fovs_counts_int)]
                
                if len(chunk_filtered) > 0:
                    # Convert to PyArrow table
                    pa_table = pa.Table.from_pandas(chunk_filtered, preserve_index=False)
                    
                    if first_chunk:
                        # Initialize parquet writer with schema from first chunk
                        parquet_writer = pq.ParquetWriter(
                            parquet_path,
                            pa_table.schema,
                            compression='snappy',
                        )
                        first_chunk = False
                    
                    # Write chunk directly to parquet
                    parquet_writer.write_table(pa_table)
                    total_rows_written += len(chunk_filtered)
                    del chunk_filtered, pa_table  # Free memory immediately
            
            # Close parquet writer
            if parquet_writer is not None:
                parquet_writer.close()
                logger.info(f"... done ({total_rows_written:,} rows written)")
                ptable = pq.read_table(parquet_path)
            else:
                logger.warning("No transcripts found for the specified FOVs.")
                ptable = None
            
            if ptable is not None:
                for fov in fovs_counts:
                    aff = affine_transforms_to_global[fov]
                    sub_table = ptable.filter(pa.compute.equal(ptable.column(CosmxKeys.FOV), int(fov))).to_pandas()
                    sub_table[CosmxKeys.INSTANCE_KEY] = sub_table[CosmxKeys.INSTANCE_KEY].astype("category")
                    # we rename z because we want to treat the data as 2d
                    sub_table.rename(columns={"z": "z_raw"}, inplace=True)
                    if "CellComp" in sub_table:
                        sub_table['CellComp'] = sub_table['CellComp'].fillna('0').astype("category")

                    if len(sub_table) > 0:
                        points[f"{fov}_points"] = PointsModel.parse(
                            sub_table,
                            coordinates={"x": CosmxKeys.X_LOCAL_TRANSCRIPT, "y": CosmxKeys.Y_LOCAL_TRANSCRIPT},
                            feature_key=CosmxKeys.TARGET_OF_TRANSCRIPT,
                            instance_key=CosmxKeys.INSTANCE_KEY,
                            transformations={
                                fov: Identity(),
                                "global": aff,
                                "global_only_labels": aff,
                            },
                        )

    # TODO: what to do with fov file?
    # if fov_file is not None:
    #     fov_positions = pd.read_csv(path / fov_file, header=0, index_col=CosmxKeys.FOV)
    #     for fov, row in fov_positions.iterrows():
    #         try:
    #             adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
    #         except KeyError:
    #             logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
    #             continue

    logger.info("Done building SpatialData object.")

    return SpatialData(images=images, labels=labels, points=points, table=table)
