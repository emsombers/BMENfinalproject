# BMENfinalproject

To download the EmoFilM dataset, please use 

# fMRI Data 
datalad install https://github.com/OpenNeuroDatasets/ds004892.git
cd ds004892
datalad get .... (the ... can be filled in with any of the specific fMRI data paths you want and they will be visible once you install)
# Annotation Data 
datalad install https://github.com/OpenNeuroDatasets/ds004872.git 
(This one should install everything because there is no large files in it)

It will not automatically donwload all the fMRI files because of their size, for ease of use we just went directly in and downloaded what we needed to use. The datalad.txt file gives you exactly what commands you can run in terminal to get the fMRI files we used. 

The general structure of this codebase is based on five Juptyer notebooks and then everything else is the data they produce. 

Notebook 1: Preprocessing includes the work done to downsample the continuous anxiety rating so they would match the temporal resolution of the fMRI files. It also involves creating various files that make the data easier to match. 
Notebook 2: Functional Connectivity includes the work necessary to create the functional connectivity matrices. It creates a folder of the ROI time series and a folder of the FC matrices along with CSVs containing their labels.
Notebook 3: Simple Machine Learning includes the work done to do some basic SVM and Random Forest classification and regression for the FC matrices. These are used to compare with the neural networks. 
Notebook 4: CNN Regression
Notebook 5: Classification Model 

Test_SingleSubjectfMRI was not used in the final project, but it is a good explanation Emma put together of how to process fMRI data into functional connectivity matrices and why we do it the way we do it in this project. 