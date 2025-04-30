## this needs to be trashed or reworked 


import nibabel as nib
import numpy as np 
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets 

# load and preprocess data for a single subject
def load_fmri_data(fmri_path):
    img = nib.load(fmri_path)
    data = img.get_fdata()
    tr = img.header.get_zooms()[3]
    affine = img.affine
    return data, tr, affine


# process a single subject and extract chunks based on events
def process_subject(fmri_path, events):
    print(f"Processing fMRI file: {fmri_path}")
    print(f"Events for this subject: {events}")
    
    data, tr, affine = load_fmri_data(fmri_path)
    extracted_chunks = []

    # Ensure events is a list of event dictionaries
    for i, e in enumerate(events):
        onset = e["onset"]
        duration = e["duration"]
        anxiety = e["anxiety"]
        
        start_vol = int(onset / tr)
        end_vol = int((onset + duration) / tr)

        # Extract the chunk from the data
        chunk = data[..., start_vol:end_vol]
        
        print(f"Event {i+1}: Volumes {start_vol}–{end_vol} → chunk shape: {chunk.shape} | Anxiety = {anxiety}")
        
        # Append chunk with additional information
        extracted_chunks.append({
            "chunk": chunk,
            "start_vol": start_vol,
            "end_vol": end_vol,
            "anxiety": anxiety,
            "affine": affine
        })
    
    # Sort chunks by start volume to keep temporal nature of data
    extracted_chunks = sorted(extracted_chunks, key=lambda x: x["start_vol"])

    return extracted_chunks