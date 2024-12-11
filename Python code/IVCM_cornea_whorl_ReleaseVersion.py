'''Analyze corneal nerve whorl images
Quantitative analysis of the whorl pattern in images of the central sub-basal nerve plexus, 
as described in the manuscript "Quantifying the corneal nerve whorl pattern" by M. Lapierre-Landry et al. (2024) 
Translational Vision Science & Technology December 2024, Vol.13, 11 https://doi.org/10.1167/tvst.13.12.11. 

An image of segmented nerves is used as input, and the algorithm calculates if a spiral-shape whorl is present. 
In the current implementation, we demonstrate the calculation of the seven whorl metrics, and produce accompanying 
graphs for the wide-field in vivo confocal microscopy dataset published by Lagali et al (2018) as described in 
https://doi.org/10.1038/sdata.2018.75 and available for download at https://doi.org/10.6084/m9.figshare.c.3950197 

Author: Maryse Lapierre-Landry <mxl1010@case.edu>
Created: 11th April, 2024
'''

import numpy as np # To store images as array and perform analysis
import math # Mathematical operation support
from PIL import Image # To read from .tif image files
import pandas as pd # To read input parameters from Excel file
from scipy import ndimage # To perform morphological operation on binary images
from scipy.interpolate import griddata # To perform image interpolation
import matplotlib.pyplot as plt # To generate data graphs
import cmcrameri.cm as cmc # Colormaps for graphs
from matplotlib.colors import ListedColormap #Colormaps for graphs

# ------------------------------------------
# PART 1: Load parameters and prepare image
# ------------------------------------------

# Load image and nerve skeleton (The code expects .tif files, which can be created after file conversion from NeuronJ)
# The code expects Image files to be 'sample name'.tif, e.g. "9OS_m1.tif"
# The codes expects masks files to be 'sample name'_skel.tif, e.g. "9OS_m1_skel.tif".

Sample_name="" #Insert sample name, e.g. "9OS_m1"
if Sample_name=="":
    print("You must edit the 'Sample_name' variable with a valid sample name")
 
Img_File_location="" #Insert folder path for all your image files, e.g. "C:/Users/Me/Documents/IVCM_Images/"
Skel_File_location="" #Insert folder path for all your masks files, e.g. "C:/Users/Me/Documents/IVCM_Masks/"
if Img_File_location=="":
    print("You must edit the 'Img_File_location' variable with a valid folder location")
if Skel_File_location=="":
    print("You must edit the 'Skel_File_location' variable with a valid folder location")

title_img=Img_File_location+Sample_name+".tif"
title_skel=Skel_File_location+Sample_name+"_skel.tif"

input_img = Image.open(title_img)
input_skel=Image.open(title_skel)

# Load input parameters from Excel file
param_file="" #Insert file path with all analysis parameters (original title 'Cornea_whorl_Human_allAnalysisParametersList.xlsx')
if param_file=="":
    print("You must edit the 'param_file' variable with a valid file path")

InputTable = pd.read_excel(param_file, sheet_name='Sheet1')
param=InputTable.loc[InputTable['SampleID']==Sample_name]

#Crop around the whorl center
regInt_all=np.asarray(input_img)
regInt=regInt_all[param['crop_start_x'].item()-1:param['crop_end_x'].item()-1,param['crop_start_y'].item()-1:param['crop_end_y'].item()-1]

regSkel_all=np.asarray(input_skel)
regSkel=regSkel_all[param['crop_start_x'].item()-1:param['crop_end_x'].item()-1,param['crop_start_y'].item()-1:param['crop_end_y'].item()-1]

# Input approximate location of the whorl center
whorl_coord=np.array([param['whorlX'].item(), param['whorlY'].item()]);

# Create an overlap of the IVCM image, nerve skeleton and manually selected whorl center for visualization
struct0=ndimage.generate_binary_structure(2, 1)
largeSkel=ndimage.binary_dilation((regSkel==255),structure=struct0,iterations=3)
centerpoint_img=np.zeros_like(regInt,dtype=np.uint8)
centerpoint_img[whorl_coord[0],whorl_coord[1]]=1
large_whorlcenterpoint=ndimage.binary_dilation(centerpoint_img,structure=struct0,iterations=15)

regSkel_image=np.zeros((np.size(regSkel,0),np.size(regSkel,1),3), dtype=np.uint8)
regSkel_image[:,:,0]=regInt*(largeSkel==0)+largeSkel*255
regSkel_image[:,:,1]=regInt*((largeSkel==0)*(large_whorlcenterpoint==0))+large_whorlcenterpoint*255
regSkel_image[:,:,2]=regInt*(largeSkel==0)+largeSkel*255

#Display IVCM image and nerve skeleton
pilImage = Image.fromarray(regInt)
pilImage.show()
pilImage = Image.fromarray(regSkel_image)
pilImage.show()

# -------------------------------------------------
# PART 2: Calculate vectors perpendicular to nerves
# -------------------------------------------------

# Refine position of the whorl center based on maximum distance from surrounding nerves

dist_from_nerves = ndimage.distance_transform_edt(regSkel==0)
center_dist_from_nerves=dist_from_nerves[whorl_coord[0]-25:whorl_coord[0]+25,whorl_coord[1]-25:whorl_coord[1]+25]
id_dist_center = np.unravel_index(np.argmax(center_dist_from_nerves, axis=None), center_dist_from_nerves.shape)
id_dist_center_pre=np.argwhere(center_dist_from_nerves==np.amax(center_dist_from_nerves))
id_dist_min=np.argmin(np.sqrt((id_dist_center_pre[:,0]-25)**2+(id_dist_center_pre[:,1]-25)**2))
id_dist_center=id_dist_center_pre[id_dist_min,:]
whorl_coord_precise=np.array([id_dist_center[0]+(whorl_coord[0]-25), id_dist_center[1]+(whorl_coord[1]-25)])

# Calculate 1) a patch size that will contain the whorl center and no nerve (patchDim_center). Max value:60. Min value:15

max_dist_center=np.nanmax(center_dist_from_nerves)
patchDim_center=max(15,(min(60,max_dist_center//2*2)))

# Calculate 2) a patch size that will on average contain one nerve (patchDim). Max value:70. Min value: 30

center_dist_from_nerves2=dist_from_nerves[whorl_coord[0]-200:whorl_coord[0]+199,whorl_coord[1]-200:whorl_coord[1]+199]
max_dist_surround=(np.mean(np.nanmax(center_dist_from_nerves2,0))+np.mean(np.nanmax(center_dist_from_nerves2,1)))/2

patchDim=round(max(30,(min(70,max_dist_surround//2*2))))
patchSize=np.array([patchDim,patchDim])

# List the start and end positions of all patches in x and y based on the size of patchDim and patchDim_center

x_coord_post=np.round(np.arange(whorl_coord_precise[0]+patchDim_center//2, np.size(regSkel,0)-patchSize[0], patchSize[0]))
x_coord_pre=np.round(np.arange((whorl_coord_precise[0]-(patchDim_center//2-1))%patchSize[0],whorl_coord_precise[0],patchSize[0]))
x_coord=np.concatenate((x_coord_pre,x_coord_post),0)
x_coord=x_coord.astype(int)

y_coord_post=np.round(np.arange(whorl_coord_precise[1]+patchDim_center//2, np.size(regSkel,1)-patchSize[1], patchSize[1]))
y_coord_pre=np.round(np.arange((whorl_coord_precise[1]-(patchDim_center//2-1))%patchSize[1],whorl_coord_precise[1],patchSize[1]))
y_coord=np.concatenate((y_coord_pre,y_coord_post),0)
y_coord=y_coord.astype(int)

# Remove areas of image where no IVCM data was acquired. Resize the mask to the patch grid size

regInt_BW=(regInt > 1)
regInt_patchMask=np.array(Image.fromarray(regInt_BW).resize((np.size(y_coord),np.size(x_coord)),Image.NEAREST))

# Calculate for every patch one vector which is on average perpendicular to the nerves

dilateFactor=2 # Dilation factor if nerves are sparse (many patches with no nerves)

mean_norm_angle=np.zeros((np.size(x_coord),np.size(y_coord)))

for i in range(0,np.size(x_coord)):
    for j in range(0,np.size(y_coord)):
        
        imgPatch=regSkel[x_coord[i]:x_coord[i]+patchSize[0]+1,y_coord[j]:y_coord[j]+patchSize[1]+1]
        
        if np.sum(imgPatch)/255<10:
            mean_norm_angle[i,j]=np.nan
        else:
            [dx,dy]=np.gradient(imgPatch,edge_order=1)
            all_angles_double=2*np.arctan(dy/dx)
            all_angles_sum=np.nansum(np.exp(1j*all_angles_double))
            mean_norm_angle[i,j]=np.rad2deg(np.angle(all_angles_sum/np.count_nonzero(dx+dy))/2)

        


# X and Y components of the average perpendicular vector for every patch
mean_dy=np.sin(np.deg2rad(mean_norm_angle))
mean_dx=np.cos(np.deg2rad(mean_norm_angle))



# Individual perpendicular vectors should preferentially point 1) all clockwise (or all counterclockwise) and 2) toward 
# the center of the whorl. Create a reference pattern where all vectors are clockwise (or counterclockwise) and point
# toward the center, then compare each individual vectors to the reference vector. Update the direction of the vectors
# by 180deg if necessary.

whorl_orient=param['whorl_orient'].item() #Is equal to 1 if clockwise, -1 if counterclockwise

whorl_coord_patch=np.array([np.size(x_coord_pre)-1,np.size(y_coord_pre)-1])

# Create reference whorl pattern starting with a few vectors and interpolating for the rest of the grid

Xq, Yq = np.meshgrid(range(0,np.size(mean_dx,0)),range(0,np.size(mean_dx,1)), indexing='ij')
starting_points_circle=np.array([[0,0],[0,whorl_coord_patch[1]],[0,np.size(mean_dx,1)-1],[whorl_coord_patch[0],np.size(mean_dx,1)-1],
                          [np.size(mean_dx,0)-1,0],[np.size(mean_dx,0)-1,whorl_coord_patch[1]],[whorl_coord_patch[0],0],
                          [np.size(mean_dx,0)-1,np.size(mean_dx,1)-1]])
starting_values_circleX=2*whorl_orient*np.array([[-0.707],[0],[0.707],[1],[-0.707],[0],[-1],[0.707]])
starting_values_circleY=2*whorl_orient*np.array([[0.707],[1],[0.707],[0],[-0.707],[-1],[0],[-0.707]])

starting_points_whorl=np.array([[whorl_coord_patch[0]-1,whorl_coord_patch[1]-1],[whorl_coord_patch[0]-1,whorl_coord_patch[1]+1],
                                [whorl_coord_patch[0]+1,whorl_coord_patch[1]-1],[whorl_coord_patch[0]+1,whorl_coord_patch[1]+1]])
starting_values_whorlX=np.array([[0.707],[0.707],[-0.707],[-0.707]])/4
starting_values_whorlY=np.array([[0.707],[-0.707],[0.707],[-0.707]])/4

starting_points=np.concatenate((starting_points_whorl,starting_points_circle),0)
starting_valuesX=np.concatenate((starting_values_whorlX,starting_values_circleX),0)
starting_valuesY=np.concatenate((starting_values_whorlY,starting_values_circleY),0)

VqX = griddata(starting_points, starting_valuesX, (Xq, Yq), method='linear')
VqY = griddata(starting_points, starting_valuesY, (Xq, Yq), method='linear')

VqX=VqX.reshape(np.size(VqX,0),np.size(VqX,1))
VqY=VqY.reshape(np.size(VqY,0),np.size(VqY,1))

#Display reference pattern as a quiver plot
x, y = np.meshgrid(range(0,np.size(y_coord)),range(0,np.size(x_coord)))
fig, ax = plt.subplots()
plt.imshow(mean_norm_angle, origin = 'upper')
q = ax.quiver(x, y, VqY/np.sqrt(VqY**2+VqX**2), -VqX/np.sqrt(VqY**2+VqX**2),scale=50.)
plt.title('Reference whorl '+ Sample_name, fontweight ="bold")

# Keep the perpendicular vector orientation the same, or augment by 180deg, to match the reference whorl 

for i in range(0,np.size(mean_dx,0)):
    for j in range(0,np.size(mean_dx,1)):
        
        if math.isfinite(mean_dx[i,j]) and math.isfinite(mean_dy[i,j]) and not (VqX[i,j]==0 and VqY[i,j]==0):
            orig_angle=np.arccos(np.clip((mean_dx[i,j]*VqX[i,j]+mean_dy[i,j]*VqY[i,j])/(np.sqrt(mean_dx[i,j]**2+mean_dy[i,j]**2)*np.sqrt(VqX[i,j]**2+VqY[i,j]**2)),-1,1))
            flip_angle=np.arccos(np.clip((-mean_dx[i,j]*VqX[i,j]-mean_dy[i,j]*VqY[i,j])/(np.sqrt(mean_dx[i,j]**2+mean_dy[i,j]**2)*np.sqrt(VqX[i,j]**2+VqY[i,j]**2)),-1,1))

            if flip_angle<orig_angle:
                mean_dy[i,j]=-mean_dy[i,j]
                mean_dx[i,j]=-mean_dx[i,j]


# Further corrections to the perpendicular vectors:
#   1) Detect if there is a large multi-patch area with no nerves near the whorl center. Create a mask of it
            
outer_bdry=np.ones((np.shape(mean_dx)))
outer_bdry[whorl_coord_patch[0]-2:whorl_coord_patch[0]+3,whorl_coord_patch[1]-2:whorl_coord_patch[1]+3]=0
BW_meandx_nan=(np.isfinite(mean_dx)+outer_bdry)==0
labeled_meandx_nan, num_features_meandx_nan=ndimage.label(BW_meandx_nan)

if labeled_meandx_nan[whorl_coord_patch[0],whorl_coord_patch[1]]==0:
    center_nan=np.zeros((np.shape(mean_dx)))
else:
    labeled_region=labeled_meandx_nan[whorl_coord_patch[0],whorl_coord_patch[1]]
    center_nan=(labeled_meandx_nan==labeled_region)

#   2) Interpolate missing vector values in areas with low nerve density (first for mean_dx, then for mean_dy)

mean_dx_points=np.argwhere(np.isfinite(mean_dx))
mean_dx_values=mean_dx[np.isfinite(mean_dx)]

struct1 = ndimage.generate_binary_structure(2, 2)
mean_dx_queries=np.logical_xor(ndimage.binary_dilation(np.isfinite(mean_dx),structure=struct1,iterations=dilateFactor), np.isfinite(mean_dx))

mean_dx_interp = griddata(mean_dx_points, mean_dx_values, (Xq, Yq), method='linear')
mean_dx_interp_extra = griddata(mean_dx_points, mean_dx_values, (Xq, Yq), method='nearest')

for i in range(np.size(mean_dx,0)):
    for j in range(np.size(mean_dx,1)):
        if mean_dx_queries[i,j]==True:
            if ~np.isnan(mean_dx_interp[i,j]):
                mean_dx[i,j]=mean_dx_interp[i,j]
            else:
                mean_dx[i,j]=mean_dx_interp_extra[i,j]

mean_dy_points=np.argwhere(np.isfinite(mean_dy))
mean_dy_values=mean_dy[np.isfinite(mean_dy)]

mean_dy_queries=np.logical_xor(ndimage.binary_dilation(np.isfinite(mean_dy),structure=struct1,iterations=dilateFactor), np.isfinite(mean_dy))

mean_dy_interp = griddata(mean_dy_points, mean_dy_values, (Xq, Yq), method='linear')
mean_dy_interp_extra = griddata(mean_dy_points, mean_dy_values, (Xq, Yq), method='nearest')

for i in range(np.size(mean_dy,0)):
    for j in range(np.size(mean_dy,1)):
        if mean_dy_queries[i,j]==True:
            if ~np.isnan(mean_dy_interp[i,j]):
                mean_dy[i,j]=mean_dy_interp[i,j]
            else:
                mean_dy[i,j]=mean_dy_interp_extra[i,j]


#   3) Making sure all vectors are at least of norm = 1

for i in range(np.size(mean_dx,0)):
    for j in range(np.size(mean_dx,1)):
        if np.abs(mean_dx[i,j])<=0.5 and np.abs(mean_dy[i,j])<=0.5:
            mag_vec=np.sqrt(mean_dx[i,j]**2+mean_dy[i,j]**2)
            mean_dx[i,j]=1*(mean_dx[i,j]/mag_vec)
            mean_dy[i,j]=1*(mean_dy[i,j]/mag_vec)

#   4) Remove any vector that might have been computer at the whorl center 

mean_dx[whorl_coord_patch[0],whorl_coord_patch[1]]=np.nan
mean_dy[whorl_coord_patch[0],whorl_coord_patch[1]]=np.nan

#   5) Using the mask created in 1), remove any vector near the whorl center if there was a large area with no nerves

mean_dx[center_nan==1]=np.nan
mean_dy[center_nan==1]=np.nan

# Display the perpendicular vectors to the nerves
            
fig1, ax1 = plt.subplots()
plt.imshow(mean_norm_angle, origin = 'upper')
q = ax1.quiver(x, y, mean_dy, -mean_dx,scale=40.,width=0.005,headwidth=3.5,headlength=3,headaxislength=2.5)
plt.title('Perpendicular vectors '+ Sample_name, fontweight ="bold")

# -------------------------------------------------
# PART 3: Connect perpendicular vectors into traces
# -------------------------------------------------

# For each position (i,j) is the vector field formed by mean_dx and mean_dy, find which neighboring patch it is
# pointing toward, and follow from one patch to the next until "traces" are linking all vectors in the vector field

mean_dx[np.isnan(mean_dx)]=0
mean_dy[np.isnan(mean_dy)]=0

# Initialize arrays and lists for the recursive function
traced_patches=np.zeros_like(mean_dx)
traced_patches_list=[]
last_patch_list=[]

# Declare the recursive function "vortex_forward_trace" to recursively link patches pointing to each other and create "traces"

def vortex_forward_trace(current_i, current_j, mean_dx, mean_dy, visited_patches):
    # This function finds which neighbor the current patch is pointing to,
    # and keeps a list of already visited nodes.

    # 1) Add the current patch to the list of patches already visited
    visited_patches.append((current_i,current_j))

    # 2) Find the patch the current patch is pointing toward
    neigh_x_coord=round(mean_dx[current_i,current_j]+current_i)
    neigh_y_coord=round(mean_dy[current_i,current_j]+current_j)

    # 3) Check if the next patch is within the boundary of the current image
    if neigh_x_coord<np.size(mean_dx,0) and neigh_x_coord>=0 and neigh_y_coord<np.size(mean_dx,1) and neigh_y_coord>=0:

        # 4) Check if we have already visited the next patch
        next_patch=(neigh_x_coord,neigh_y_coord)
        if not (next_patch in visited_patches):

            # 5) Then call the function recursively on the next patch
            visited_patches=vortex_forward_trace(neigh_x_coord,neigh_y_coord,mean_dx, mean_dy,visited_patches)

    return visited_patches

# Call the recursive function on all patches where nerves are present 

for i in range(np.size(mean_dx,0)):
    for j in range(np.size(mean_dx,1)):

        if np.isfinite(mean_norm_angle[i,j]):
            visited_patches=[]
            visited_patches=vortex_forward_trace(i,j,mean_dx,mean_dy,visited_patches)
            last_patch_list.append(visited_patches[-1])
            traced_patches_list.append(visited_patches)


# Create endpoint diagram
endpoints_map=np.zeros_like(mean_dx)

for i in range(len(last_patch_list)):
    endpoints_map[last_patch_list[i]]=endpoints_map[last_patch_list[i]]+1

# Display endpoint diagram
fig3, ax3 = plt.subplots()
endpoints_mapGraph=endpoints_map.copy()
endpoints_mapGraph[endpoints_mapGraph==0]=np.nan
ed=plt.imshow(endpoints_mapGraph,cmap=cmc.batlow_r)
plt.title('Endpoint Diagram '+ Sample_name, fontweight ="bold")
cbar = fig3.colorbar(ed)

# -------------------------------------------------------------------
# PART 4: Classify endpoints into "Center", "Off-center" and "Image edges"
# -------------------------------------------------------------------
region_map=np.zeros_like(mean_dx)

# List all endpoints coordinates (only unique values)
unique_endpoints=list(set(last_patch_list))
unique_endpoints_array=np.asarray(unique_endpoints)

# If an endpoint is at the edge of the image (or at the edge of the acquired FOV as determined by regInt_patchMask)
# indicate as 0.5 in region_map
for i in range(len(unique_endpoints)):
    if regInt_patchMask[unique_endpoints[i]]==0:
        region_map[unique_endpoints[i]]=0.5

    if (unique_endpoints_array[i,0] == 0 or unique_endpoints_array[i,1] == 0 or 
            unique_endpoints_array[i,0]==np.size(mean_dx,0)-1 or unique_endpoints_array[i,1]==np.size(mean_dx,1)-1):
        region_map[unique_endpoints[i]]=0.5

#If an endpoint is to be considered the whorl "center", indicate as 1 in region_map
# Center need to be 1) Near the user-selected centerpoint, 2) account for >10% of the total area

#Find candidate center points which are near the whorl center
cc_centerpoints, num_features_endpoint_map=ndimage.label(endpoints_map>0,structure=struct1)
candidate_cc=np.unique(cc_centerpoints[whorl_coord_patch[0]-1:whorl_coord_patch[0]+2,whorl_coord_patch[1]-1:whorl_coord_patch[1]+2])
candidate_cc=candidate_cc[candidate_cc!=0]

#Calculate the area of each of the candidates
maj_center=[]

for i in range(len(candidate_cc)):
    maj_center.append(np.sum(endpoints_map[cc_centerpoints==candidate_cc[i]])/len(last_patch_list))

#Find the maximum area
max_maj_center = max(maj_center)
id_maj_center = maj_center.index(max_maj_center)

#If the max area is at least 10% of the total area, identify as a center endpoint
if maj_center[id_maj_center]>=0.1:
    region_map[cc_centerpoints==candidate_cc[id_maj_center]]=1


# -----------------------------------------------------------------------------------
# PART 5: Calculate whorl metrics: whorl area, whorl standard deviation, and endpoint isotropic score
# -----------------------------------------------------------------------------------

#The whorl area is the total image area (in units of pixel^2) linked via a trace to a center endpoint (the whorl center)
#Regular whorl should have high areas, if no whorl is present, the area will be low or zero
whorl_area=np.sum(endpoints_map[region_map==1])*patchDim**2

#The whorl standard deviation (in units of pixel) is based on the distance between each non-edge endpoints and 
#the whorl center. Endpoints concentrated near the center lead to a small st.dev, endpoints dispersed far from the
#center lead to a large st.dev. Endpoints at the edge of the image are excluded from the calculation since they 
#occur mostly from the cropping of the image, and not from the shape of the whorl pattern

noedge_endpoints_map=endpoints_map*(region_map!=0.5)*(patchDim**2)
noedge_endpoints_map_idx=np.argwhere(noedge_endpoints_map)
value_endTrace=[]
variance_item=[]


for i in range(len(noedge_endpoints_map_idx)):
    noedge_endpoints_coord=noedge_endpoints_map_idx[i]
    value_endTrace.append(noedge_endpoints_map[noedge_endpoints_coord[0],noedge_endpoints_coord[1]])
    variance_item.append(value_endTrace[i]*((whorl_coord_patch[0]*patchDim-noedge_endpoints_coord[0]*patchDim)**2+
    (whorl_coord_patch[1]*patchDim-noedge_endpoints_coord[1]*patchDim)**2))

whorl_std=math.sqrt(sum(variance_item)/(sum(value_endTrace)-1))

# Display endpoint diagram without any edge points (with whorl area and whorl st.dev displayed)
fig4, ax4 = plt.subplots()
endpoints_noedge_graph=endpoints_map*(region_map!=0.5)*(patchDim**2)
endpoints_noedge_graph[endpoints_noedge_graph==0]=np.nan
ed=plt.imshow(endpoints_noedge_graph,cmap=cmc.batlow_r)
plt.title('Endpoint Diagram (no edge-endpoints) for '+ Sample_name + '\n Area: ' + str(round(whorl_area)) 
          + 'pix$^2$ St.Dev: '+ str(round(whorl_std,2)) + 'pix')
cbar = fig4.colorbar(ed)

#The endpoint isotropic score is created to detect one phenomenom: a "ring" of endpoints around the whorl center. This
#configuration of endpoints most often happens in web-like whorl pattern, when traces form an almost perfect circle until
#pointing back to themselves. When not taken into account, a "ring" of endpoints would severly affect the results of the 
#whorl area and whorl st.dev

#Divide images in 45deg pie slices to count the endpoints present in each
pieSlice_image=np.zeros_like(mean_dx)

for i in range(np.size(pieSlice_image,0)):
    for j in range(np.size(pieSlice_image,1)):
        if i<=whorl_coord_patch[0]:
            if j<whorl_coord_patch[1]:
                if (j-i)>=whorl_coord_patch[1]-whorl_coord_patch[0]:
                    pieSlice_image[i,j]=1
                else: pieSlice_image[i,j]=2
            else:
                if j-whorl_coord_patch[1]<whorl_coord_patch[0]-i:
                    pieSlice_image[i,j]=8
                else: pieSlice_image[i,j]=7
        else:
            if j<whorl_coord_patch[1]:
                if whorl_coord_patch[1]-j>i-whorl_coord_patch[0]:
                    pieSlice_image[i,j]=3
                else: pieSlice_image[i,j]=4
            else:
                if j-whorl_coord_patch[1]<i-whorl_coord_patch[0]:
                    pieSlice_image[i,j]=5
                else: pieSlice_image[i,j]=6

#Calculate the proportion of endpoints in each 45deg sections (excluding endpoints at the image edges
# and endpoints at the whorl center) 
 
middle_endpoints_map=(region_map==0)*(endpoints_map>0)
pieSlice_endpointProp=[]
for i in range(1,9):
    pieSlice_endpointProp.append(np.sum(middle_endpoints_map[pieSlice_image==i])/np.sum(pieSlice_image==i))

#Calculate the endpoint isotropic score as the coefficient of variance (can be NaN if there are no endpoints)
endpoint_iso_score=np.std(pieSlice_endpointProp)/np.mean(pieSlice_endpointProp)

# ----------------------------------------------------------
# PART 6: Calculate whorl metrics: nerve density (around the whorl center)
# ----------------------------------------------------------

center_idStartX=max(0,whorl_coord[0]-349)
center_idStartY=max(0,whorl_coord[1]-349)
center_idEndX=min(np.size(regInt,0)-1,whorl_coord[0]+350)
center_idEndY=min(np.size(regInt,0)-1,whorl_coord[0]+350)

#Portion of the nerve skeleton around the whorl center
centerSkel=regSkel[center_idStartX:center_idEndX+1,center_idStartY:center_idEndY]

#Portion of the image field-of-view around the center
center_regIntBW=regInt_BW[center_idStartX:center_idEndX+1,center_idStartY:center_idEndY]

nerve_density=100*np.sum(centerSkel/255)/np.sum(center_regIntBW)

# ----------------------------------------------------------------------------------------
# PART 7: Calculate whorl metrics: spiral fill score, spiral isotropic score, spiral score
# ----------------------------------------------------------------------------------------

#From the endpoints at the whorl center, follow each trace for n=5 patches. Observing the resulting sub-group of patches
#determine if the whorl is large and/or symmetric around the center. 

#Locate all patches that are adjacent to the previously identified center endpoints. 
center_adjacent=np.zeros_like(mean_dx)
center_adjacent[region_map==1]=1
center_adjacent=ndimage.binary_dilation(center_adjacent,structure=struct1)
idx_centerAdjacent=np.argwhere(center_adjacent)
idx_centerAdjacent_list=list(map(tuple,idx_centerAdjacent))

#List of center endpoints
center_points=np.argwhere(region_map==1)
center_points_list=list(map(tuple,center_points))

#Step out from the center endpoints outward (do it for each trace that ends at the center)
trace_stepOut=np.zeros_like(mean_dx)

for ii in range(len(traced_patches_list)):
    #If the current trace ends at one of the center endpoints
    if len(set(center_points_list).intersection(traced_patches_list[ii]))!=0:

        #List of patches which are part of the current trace
        trace_id=list(reversed(traced_patches_list[ii]))

        #Find which patch in the trace is part of idx_centerAdjacent (center or adjacent to center)
        idx_last_intersect=len(trace_id)-1

        for i in range(len(trace_id)):
            if (trace_id[i] in idx_centerAdjacent_list):
                idx_last_intersect=i
            else: break

        trace_up_to_five_steps=trace_id[0:min(len(trace_id),idx_last_intersect+6)]
        
        for id in trace_up_to_five_steps:
            trace_stepOut[id]=1

trace_stepOut[region_map==1]=2  

# Compare trace_stepOut to a dilation of the center endpoints for 5 patches, i.e., an ideal isotropic step-out,
# which could be the result of a perfect spiral with straight traces leaving the center equally in all directions

# Use dilation to create the ideal step-out.
struct2 =np.ones((11,11))
struct2[0,0]=0; struct2[0,1]=0; struct2[1,0]=0; struct2[9,0]=0; struct2[10,0]=0; struct2[10,1]=0; 
struct2[0,9]=0; struct2[0,10]=0; struct2[1,10]=0; struct2[9,10]=0; struct2[10,10]=0; struct2[10,9]=0

ideal_stepOut=ndimage.binary_dilation(center_adjacent,structure=struct2,iterations=1)

# Calculate metric: spiral fill score
whorl_spiralFill=10-10*np.sum(ideal_stepOut-np.clip(trace_stepOut,0,1))/np.sum(ideal_stepOut)

# To calculate the spiral isotropic score, determine if the trace_stepOut is radially symmetric around the center
# First divide the image into 45deg pie slices, then calculate the coefficient of variance of the area in each 45deg slice

idx_ideal=np.argwhere(ideal_stepOut)
row_ideal=idx_ideal[:,0]
col_ideal=idx_ideal[:,1]

idealStepOut_pieSlice=np.zeros_like(mean_dx)

# Create "ideal" 45deg each pie slices. The center of the pie should be the middle (centroid) of the center endpoints
centroid_centerEndpoints=ndimage.center_of_mass(center_adjacent)

for i in range(np.size(idx_ideal,0)):
    if row_ideal[i]<=centroid_centerEndpoints[0]:
        if col_ideal[i]<centroid_centerEndpoints[1]:
            if col_ideal[i]-row_ideal[i]>=(centroid_centerEndpoints[1]-centroid_centerEndpoints[0]):
                idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=1
            else: idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=2
        else:
            if col_ideal[i]-centroid_centerEndpoints[1]<centroid_centerEndpoints[0]-row_ideal[i]:
                idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=8
            else: idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=7
    else:
        if col_ideal[i]<centroid_centerEndpoints[1]:
            if centroid_centerEndpoints[1]-col_ideal[i]>row_ideal[i]-centroid_centerEndpoints[0]:
                idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=3
            else: idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=4
        else:
            if col_ideal[i]-centroid_centerEndpoints[1]<row_ideal[i]-centroid_centerEndpoints[0]:
                idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=5
            else: idealStepOut_pieSlice[row_ideal[i],col_ideal[i]]=6

# Proportion of trace_stepOut inside each of the 45deg pie slices

trace_stepOut_map=np.clip(trace_stepOut,0,1)
pieSlice_stepOutProp=[]
for i in range(1,9):
    pieSlice_stepOutProp.append(np.sum(trace_stepOut_map[idealStepOut_pieSlice==i])/np.sum(idealStepOut_pieSlice==i))

#Calculate whorl metric: spiral isotropic score as the coefficient of variance
whorl_spiralIso=np.std(pieSlice_stepOutProp)/np.mean(pieSlice_stepOutProp)


#Display the step-out operation with spiral fill score and spiral isotropic score 
fig6, ax6 = plt.subplots()
plt.imshow(trace_stepOut+ideal_stepOut,cmap=cmc.oslo_r)
plt.title('Step-out Diagram (Five steps from center) for '+ Sample_name + '\n Spiral Fill: ' + str(round(whorl_spiralFill,2)) 
          + '/10 Sprial Iso.: '+ str(round(whorl_spiralIso,2)))


# ----------------------------------------------------------------------------------------
# PART 8: Display traces 1) All traces, 2) Only traces that end at the whorl center
# ----------------------------------------------------------------------------------------

# Display all traces and endpoints over an image of the nerves

fig7, ax7 = plt.subplots()
cmap = ListedColormap(["black", "fuchsia"])
im=ax7.imshow(largeSkel,vmin=0, vmax=1,cmap=cmap)

for i in range(len(traced_patches_list)):
    trace_idx=np.array(traced_patches_list[i])

    row_trace_res=np.zeros((np.size(trace_idx,0),1))
    col_trace_res=np.zeros((np.size(trace_idx,0),1))

    for j in range(np.size(trace_idx,0)):
        if trace_idx[j,0]==len(x_coord)-1:
            row_trace_res[j]=x_coord[-1]+patchDim/2
        else:row_trace_res[j]=(x_coord[trace_idx[j,0]]+x_coord[trace_idx[j,0]+1])/2  

    for j in range(np.size(trace_idx,0)):
        if trace_idx[j,1]==len(y_coord)-1:
            col_trace_res[j]=y_coord[-1]+patchDim/2
        else:col_trace_res[j]=(y_coord[trace_idx[j,1]]+y_coord[trace_idx[j,1]+1])/2

    ax7.plot(col_trace_res,row_trace_res,linewidth=1, color='white')
    ax7.plot(col_trace_res[-1],row_trace_res[-1],'*y',ms=10)

plt.title('All nerves (pink) with traces (white) and endpoints (yellow) for '+ Sample_name)

# Display only the traces that end at the center endpoints, over an image of the nerves

fig8, ax8 = plt.subplots()
im2=ax8.imshow(largeSkel,vmin=0, vmax=1,cmap=cmap)

for i in range(len(traced_patches_list)):
    if len(set(center_points_list).intersection(traced_patches_list[i]))!=0:
        trace_idx=np.array(traced_patches_list[i])

        row_trace_res=np.zeros((np.size(trace_idx,0),1))
        col_trace_res=np.zeros((np.size(trace_idx,0),1))

        for j in range(np.size(trace_idx,0)):
            if trace_idx[j,0]==len(x_coord)-1:
                row_trace_res[j]=x_coord[-1]+patchDim/2
            else:row_trace_res[j]=(x_coord[trace_idx[j,0]]+x_coord[trace_idx[j,0]+1])/2  

        for j in range(np.size(trace_idx,0)):
            if trace_idx[j,1]==len(y_coord)-1:
                col_trace_res[j]=y_coord[-1]+patchDim/2
            else:col_trace_res[j]=(y_coord[trace_idx[j,1]]+y_coord[trace_idx[j,1]+1])/2

        ax8.plot(col_trace_res,row_trace_res,linewidth=1, color='white')
        ax8.plot(col_trace_res[-1],row_trace_res[-1],'*y',ms=10) 

plt.title('Nerves (pink) with only the traces (white) that end at the \n center endpoints (yellow) for '+ Sample_name)
plt.show()

