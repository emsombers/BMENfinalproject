### this can probably be trashed 


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
def process_subject(fmri_path, events, low_threshold=33, high_threshold=66):
    print(f"Processing fMRI file: {fmri_path}")
    print(f"Events for this subject: {events}")
    
    data, tr, affine = load_fmri_data(fmri_path)
    extracted_chunks = []

    # Ensure events is a list of event dictionaries
    for event in events:
        print(f"Processing event: {event}")  # Print the individual event
        onset = event["onset"]
        duration = event["duration"]
        anxiety = event["anxiety"]
        
        start_vol = int(onset / tr)
        end_vol = int((onset + duration) / tr)

        chunk = data[..., start_vol:end_vol]
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
    low_anx_chunks = [chunk for chunk in extracted_chunks if chunk["anxiety"] <= low_threshold]
    high_anx_chunks = [chunk for chunk in extracted_chunks if chunk["anxiety"] >= high_threshold]

    return low_anx_chunks, high_anx_chunks, tr, affine


# compute FC for a given chunk
def compute_fc(chunk_4d, tr, affine, masker):
    chunk_img = nib.Nifti1Image(chunk_4d, affine)
    roi_ts = masker.fit_transform(chunk_img)
    fc_matrix = ConnectivityMeasure(kind="correlation").fit_transform([roi_ts])[0]
    return fc_matrix


# compute FC for low and high anxiety groups based on anxiety score thresholds
def compute_fc_for_anxiety_groups(low_anx_chunks, high_anx_chunks, tr, affine, masker):
    """
    Computes the FC matrices for both low and high anxiety groups by averaging across subjects' chunks
    after binning by anxiety score.
    """
    fc_low = np.mean([compute_fc(chunk["chunk"], tr, affine, masker) for chunk in low_anx_chunks], axis=0) if low_anx_chunks else None
    fc_high = np.mean([compute_fc(chunk["chunk"], tr, affine, masker) for chunk in high_anx_chunks], axis=0) if high_anx_chunks else None
    
    return fc_low, fc_high


# compute FC for low and high anxiety groups by using the network groupings
def compute_grouped_fc(chunk_4d, tr, affine, network_groupings, masker):
    """
    Computes FC matrices based on network groupings (e.g., Visual, Limbic).
    """
    chunk_img = nib.Nifti1Image(chunk_4d, affine)
    roi_ts = masker.fit_transform(chunk_img)
    
    # Initialize the FC matrix for grouped networks
    fc_matrix_grouped = np.zeros((len(network_groupings), len(network_groupings)))
    
    # Loop through each pair of networks
    for i, (network_1, regions_1) in enumerate(network_groupings.items()):
        for j, (network_2, regions_2) in enumerate(network_groupings.items()):
            ts_1 = np.mean(roi_ts[:, regions_1], axis=1) if regions_1 else np.zeros(roi_ts.shape[0])
            ts_2 = np.mean(roi_ts[:, regions_2], axis=1) if regions_2 else np.zeros(roi_ts.shape[0])
            
            # Compute correlation between the time series of two networks
            fc_matrix_grouped[i, j] = np.corrcoef(ts_1, ts_2)[0, 1]
    
    return fc_matrix_grouped


# 6. Aggregate FC for low and high anxiety across multiple subjects
def compute_fc_across_subjects(subjects_data, masker):
    all_fc_low = []
    all_fc_high = []
    all_low_anx = []
    all_high_anx = []

    # Iterate over each subject
    for subject_data in subjects_data:
        # Compute FC for the current subject
        fc_low, fc_high, low_anx, high_anx = compute_fc_for_anxiety_groups(subject_data, masker)
        
        # Store FC matrices and anxiety scores
        all_fc_low.append(fc_low)
        all_fc_high.append(fc_high)
        all_low_anx.append(low_anx)
        all_high_anx.append(high_anx)

    # Aggregate across subjects (e.g., average FC matrices)
    avg_fc_low = np.mean(all_fc_low, axis=0)
    avg_fc_high = np.mean(all_fc_high, axis=0)

    # Aggregate anxiety scores (optional, depending on what you need)
    avg_low_anx = np.mean(all_low_anx)
    avg_high_anx = np.mean(all_high_anx)

    return avg_fc_low, avg_fc_high, avg_low_anx, avg_high_anx


# 7. Main function to process multiple subjects
def main(subjects_fmri_paths, subjects_events, masker, low_threshold=33, high_threshold=66):
    all_fc_low = []
    all_fc_high = []

    # Process each subject and compute FC for anxiety groups
    for fmri_path, events in zip(subjects_fmri_paths, subjects_events):
        low_anx_chunks, high_anx_chunks, tr, affine = process_subject(fmri_path, events, masker, low_threshold, high_threshold)
        
        # Compute FC for low and high anxiety groups for each subject
        fc_low, fc_high = compute_fc_for_anxiety_groups(low_anx_chunks, high_anx_chunks, tr, affine, masker)
        
        # Store the results for aggregation
        if fc_low is not None:
            all_fc_low.append(fc_low)
        if fc_high is not None:
            all_fc_high.append(fc_high)

    # Aggregate FC matrices (e.g., average across subjects)
    avg_fc_low = np.mean(all_fc_low, axis=0) if all_fc_low else None
    avg_fc_high = np.mean(all_fc_high, axis=0) if all_fc_high else None
    
    # Print results (or return them)
    print(f"Average Low Anxiety FC: {avg_fc_low}")
    print(f"Average High Anxiety FC: {avg_fc_high}")

