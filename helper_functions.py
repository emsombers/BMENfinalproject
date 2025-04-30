import nibabel as nib
import json
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets 

def process_fmri_data(fmri_path, event_data):
    """
    Process the fMRI data and extract anxiety chunks based on the events provided.
    
    Args:
        fmri_path (str): Path to the fMRI data file.
        event_data (dict): A dictionary containing event information, such as onset, duration, and anxiety levels.
        
    Returns:
        extracted_chunks (list): A list of dictionaries containing extracted fMRI chunks and their associated data.
        low_anx_chunks (list): List of low anxiety chunks.
        high_anx_chunks (list): List of high anxiety chunks.
    """
    # Load fMRI data
    img = nib.load(fmri_path)
    data = img.get_fdata()
    tr = img.header.get_zooms()[3]
    print(tr)
    affine = img.affine
    print(f"fMRI data loaded. Data shape: {data.shape}")

    extracted_chunks = []

    # Process events and extract anxiety chunks
    for i, e in enumerate(event_data):
        onset = e["onset"]
        duration = e["duration"]
        anxiety = e["anxiety"]

        start_vol = int(onset / tr)
        end_vol = int((onset + duration) / tr)

        chunk = data[..., start_vol:end_vol]
        print(f"Event {i+1}: Volumes {start_vol}–{end_vol} → chunk shape: {chunk.shape} | Anxiety = {anxiety}")

        extracted_chunks.append({
            "chunk": chunk,
            "start_vol": start_vol,
            "end_vol": end_vol,
            "anxiety": anxiety,
            "affine": affine
        })
    
    # Sort chunks by start volume to keep temporal nature of data
    extracted_chunks = sorted(extracted_chunks, key=lambda x: x["start_vol"])
    
    # Now bin the chunks based on anxiety scores (low vs high)
    low_anx_chunks = [chunk for chunk in extracted_chunks if chunk["anxiety"] <= 33]
    high_anx_chunks = [chunk for chunk in extracted_chunks if chunk["anxiety"] >= 66]

    print(f"Number of low anxiety chunks: {len(low_anx_chunks)}")
    print(f"Number of high anxiety chunks: {len(high_anx_chunks)}")

    return extracted_chunks, low_anx_chunks, high_anx_chunks 

