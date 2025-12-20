"""
Utility functions for downloading data from HuggingFace repositories.
"""
import os
from typing import Optional
from huggingface_hub import hf_hub_download


def download_from_hf(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    repo_type: str = "dataset"
) -> str:
    """
    Download a file from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID 
                 (e.g., "WatermelonAnh/WattiesMammals" for datasets,
                  "WatermelonAnh/Watties_Data" for backbone/SMAL data)
        filename: Path to file in repository (e.g., "data/smal/my_smpl_00781_4_all.pkl")
        token: HuggingFace token (optional)
        cache_dir: Cache directory (optional)
        repo_type: Repository type ("dataset" or "model")
    
    Returns:
        Local path to downloaded file
    """
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            # token=token,
            cache_dir=cache_dir,
            # repo_type=repo_type
        )
        return local_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {filename} from HuggingFace repository {repo_id}. "
            f"Error: {str(e)}"
        )


def download_smal_model(
    repo_id: str,
    filename: str = "data/smal/my_smpl_00781_4_all.pkl",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download SMAL model file from HuggingFace.
    
    Note: SMAL data is stored in WatermelonAnh/Watties_Data repository.
    
    Args:
        repo_id: HuggingFace repository ID (should be "WatermelonAnh/Watties_Data")
        filename: Path to SMAL model file in repository
        token: HuggingFace token
        cache_dir: Cache directory
    
    Returns:
        Local path to SMAL model file
    """
    return download_from_hf(repo_id, filename, token, cache_dir)


def download_backbone_weights(
    repo_id: str,
    filename: str = "data/backbone.pth",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download backbone weights from HuggingFace.
    
    Note: Backbone weights are stored in WatermelonAnh/Watties_Data repository.
    
    Args:
        repo_id: HuggingFace repository ID (should be "WatermelonAnh/Watties_Data")
        filename: Path to backbone weights file in repository
        token: HuggingFace token
        cache_dir: Cache directory
    
    Returns:
        Local path to backbone weights file
    """
    return download_from_hf(repo_id, filename, token, cache_dir, repo_type="model")


def download_smal_prior(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Download SMAL prior file (shape or pose prior) from HuggingFace.
    
    Note: SMAL prior data is stored in WatermelonAnh/Watties_Data repository.
    
    Args:
        repo_id: HuggingFace repository ID (should be "WatermelonAnh/Watties_Data")
        filename: Path to prior file in repository
        token: HuggingFace token
        cache_dir: Cache directory
    
    Returns:
        Local path to prior file
    """
    return download_from_hf(repo_id, filename, token, cache_dir)


def resolve_path(
    path: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    is_hf_path: bool = False,
    repo_type: str = "dataset"
) -> str:
    """
    Resolve a path - either local or HuggingFace.
    
    Args:
        path: File path (local or HuggingFace relative path)
        repo_id: HuggingFace repository ID (if path is from HF)
                 Use "WatermelonAnh/Watties_Data" for backbone/SMAL data
        token: HuggingFace token
        cache_dir: Cache directory
        is_hf_path: Whether path is a HuggingFace path (not absolute/local)
        repo_type: Repository type ("dataset" or "model")
    
    Returns:
        Resolved local path
    """
    # If path is absolute or exists locally, return as-is
    if os.path.isabs(path) or os.path.exists(path):
        return path
    
    # If it's a HuggingFace path and repo_id is provided, download it
    if is_hf_path and repo_id is not None:
        return download_from_hf(repo_id, path, token, cache_dir, repo_type=repo_type)
    
    # Otherwise, assume it's a relative local path
    return path

