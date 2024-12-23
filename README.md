# Quantifying the Corneal Nerve Whorl pattern

Accompanying publication: *Lapierre-Landry, M. et al. (2024) "Quantifying the Corneal Nerve Whorl Pattern." Translational Vision Science and Technology. 13(12):11*
Link: [Paper](https://doi.org/10.1167/tvst.13.12.11)

## What is the purpose of this code?

The nerves of a specific region of the cornea form a spiral-like structure, the whorl, which is not a well-understood structure. It is different between individuals, it appears to change over time for the same eye, and it is being investigated as a marker of disease, such as diabetes and diabetic neuropathy. While observers have been describing qualitative changes to this whorl structure in studies, or have reported other metrics to characterize the nerves (such as the nerve density), we present here the first algorithm to quantify the shape of the spiral itself. With this approach, we show how to statistically compare these spiral-like patterns between experimental cohorts in larger studies of eye health. Our method can also be seen as a generalizable approach, applicable to all sorts of non-corneal, non-nerves, spiral-like patterns! 

## What does the code do?

The goal of this code is to detect if a spiral-like structure is present in a segmented nerve image. To achieve this, vectors perpendicular to each nerves are calculated. In an approach inspired by [hurricane detection](https://ieeexplore.ieee.org/abstract/document/6460709), those perpendicular vectors should point to the center of the spiral, if such a spiral is present. From vector field, seven metrics are calculated to quantify different aspects of the whorl. See publication ([link](https://doi.org/10.1167/tvst.13.12.11)) for a detailed explanation of each whorl metric.

## Demonstration of the algorithm on *in vivo* confocal microscopy images acquired in humans

In its current iteration, the code is set up to replicate the human data presented in our [publication](https://doi.org/10.1167/tvst.13.12.11). This publicly-available dataset was initially published by Lagali et al. and was acquired in subjects with type 2 diabetes and healthy controls. It can be dowloaded [here](https://doi.org/10.6084/m9.figshare.c.3950197) with all related patient data. 

Two implementations of the code are made available: 1) in MATLAB and 2) in Python. Results published in our publication were obtained from the MATLAB implementation, but we made sure the Python implementation performed the same steps and produced the same results, to be best of our abilities. 

### Code demonstration on one image (in Python)
1) Download the code file "IVCM_cornea_whorl_ReleaseVersion.py" and the "requirements.txt" file.
2) Download the "Example Data" and "Parameter list" folders
3) Install necessary packages by executing: (The '/path/to/' should be replaced by the correct folder path)
``` 
pip install -r /path/to/requirements.txt
```
4) Using an editor of your choice, edit the variables in IVCM_cornea_whorl_ReleaseVersion.py as follows:
```
Sample_name='9OS_m1'
Img_File_location='/path/to/Example data/'
Skel_File_location='/path/to/Example data/'
param_file='/path/to/Parameter list/Cornea_whorl_Human_allAnalysisParametersList.xlsx'
```
5) Run "IVCM_cornea_whorl_ReleaseVersion.py"

### To replicate results in the publication
1) Download the Lagali et al. dataset
2) Convert all images and all nerve masks to .tif
3) Download the Parameter list Excel file "Cornea_whorl_Human_allAnalysisParametersList.xlsx"
4) Install requirements.txt
5) Open the whorl-metrics code in the editor of your choice. Edit the file so that filepaths and filenames match what is on your device
6) Run the code

## Can I use this code on my own data? 

Of course! However...

### Segmented nerves
This code expects an already segmented image of the cornea nerves as input. **This code does not segment the nerves.**
Other tools should be used prior to this code to segment the nerves in an image, such as the NeuronJ plugin available for ImageJ.

### Image type
The current code implementation expects both a .tif image of the cornea, and a .tif binary mask of the segmented nerves. The code can easily be edited to accomodate other formats. 

### Create your own Parameter List file
You will need to create a file similar to the current "Cornea_whorl_Human_allAnalysisParametersList.xlsx" which indicates how your original image should be cropped (x and y start/end coordinates), the approximate x and y coordinates of the whorl center, and the approximate orientation of the whorl (from the outside-in, the whorl is clockwise = -1, or counterclockwise = 1). 






