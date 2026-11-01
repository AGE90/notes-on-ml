"""
Data loading utilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ml.utils.paths import data_external_dir

PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3"
PTBXL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load data from a CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_csv(filepath, **kwargs)


def load_excel(
    filepath: str | Path, sheet_name: str | int | None = 0, **kwargs
) -> pd.DataFrame:
    """Load data from an Excel file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the Excel file
    sheet_name : Optional[Union[str, int]], optional
        Name or index of the sheet to load, by default 0

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)


def load_parquet(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load data from a Parquet file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the Parquet file

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_parquet(filepath, **kwargs)


def load_numpy(filepath: str | Path, **kwargs) -> np.ndarray:
    """Load data from a NumPy file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the NumPy file

    Returns
    -------
    np.ndarray
        Loaded data
    """

    data: np.ndarray = np.load(filepath, **kwargs)
    return data


def load_ptbxl_ecg(record: str = "00001") -> np.ndarray:
    """Load a 12-lead, 10 s, 500 Hz ECG from PTB-XL (PhysioNet, CC BY 4.0).

    The record is downloaded once to `data/external/ptbxl/`. The file holds 16-bit
    samples at 1000 units per mV, interleaved by lead (WFDB format 16).

    Parameters
    ----------
    record : str, optional
        Five-digit PTB-XL record id, by default "00001"

    Returns
    -------
    np.ndarray
        Signal in mV, shape (5000, 12); leads ordered as `PTBXL_LEADS`.
    """
    path = data_external_dir("ptbxl", f"{record}_hr.dat")
    if not path.exists():
        folder = int(record) // 1000 * 1000
        url = f"{PTBXL_URL}/records500/{folder:05d}/{record}_hr.dat"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return np.fromfile(path, dtype="<i2").reshape(-1, 12) / 1000
