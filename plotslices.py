#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:26:11 2023

@author: triloo
"""
# Import tools
import nilearn.plotting
import nilearn.image   
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
import nibabel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    
def plotslices(brainval: np.ndarray, atlas4D: str, structural: str ='/opt/fsl/6.0.1/data/standard/MNI152_T1_1mm_brain.nii.gz', 
               saveloc: str= None, orientation: str ='z', cut_coords: int =5, lineW: float = 1, colors='Greys'):
    
    """  
    Plotting ROIs in a brain using contour mapping from nilearn. Values in brainval
    scale ROI visualisation alpha in the plot.
    
    Parameters:
    brainval:     numeric vector of length number of ROIs. Values not negative
    atlas4D:      string adress to binarized 4D atlas to use with ROIs
    structural    string adress to structural MRI for background 
    saveloc:      string adress to save the image
    orientation:  string xyz axis orientation of brain slices
    cut_coords:   numeric number of cuts or vector of cut coordinates
    lineW:        line width for contours
    colors:       array 1 x RGB colors, either one color or an array of len(brainval)    
    
    Returns:
    brainfig: nilearn.plotting.Displays - Generated brain visualization

    """   
    

    #defensive 
    #check if any brainval value < 0 
    if np.any(brainval<0):
        raise ValueError('brainval values should not be negative')
    
    #scale brainval between 0 and 1
    brainval = np.divide(brainval, np.max(brainval))
    
    # Colors
    initialcol='Greys'
    
    # Takes first color entry if not len(brainval) = len(colors)
    if len(brainval) == len(colors):
        pal =  colors
    else:
        pal=np.array([colors[0] for i in range(len(brainval))])
    
    
    # Load brain data and extract to numeric
    atlas = nibabel.load(atlas4D)
    atlas_dat=atlas.get_fdata(dtype=np.float32).T    
    
    # Clean regions we don't want to plot
    atlas_dat[brainval == 0] = 0
            
    # Mask for our combined ROIs
    atlas=nibabel.nifti1.Nifti1Image(atlas_dat.T,atlas.affine)
    atlas_mask = nibabel.nifti1.Nifti1Image(np.sign(np.sum(atlas_dat,0)).T,atlas.affine)

    
    # Initiate plot object
    brainfig=nilearn.plotting.plot_roi(atlas_mask, atlas_mask, colorbar=False, cut_coords=cut_coords, display_mode=orientation, alpha=1, cmap=initialcol, draw_cross=False,black_bg=False,annotate=False)
    
    # Create iterable for ROIs
    atlas_imgs=nilearn.image.iter_img(atlas)

    # Draw the ROIs in plot object according to provided brainval alphas and colors
    for j,img in enumerate(atlas_imgs):
        mean_epi = mean_img(img)
            
        img2=nilearn.image.smooth_img(mean_epi,1)
        brainfig.add_contours(img2,filled=True,levels=[0.02],cmap=None,colors=(pal[j],pal[j]),linewidths=lineW,alpha=brainval[j])
        
    # Add contours from structural image for background
    brainfig.add_contours(nilearn.image.smooth_img(structural, 5), alpha=1, levels=[95], linewidths=lineW, cmap=sns.dark_palette('w', as_cmap=True),)     
    brainfig.add_contours(nilearn.image.smooth_img(structural, 0.5), alpha=0.8, levels=[5000], linewidths=lineW, cmap=sns.dark_palette('w',as_cmap=True))
    
    # Annotate brain coordinates per slice
    brainfig.annotate(left_right=False,size=int(12*lineW))   
    
    
    # Save at saveloc
    if saveloc:
        brainfig.savefig(saveloc) 
        
    return brainfig