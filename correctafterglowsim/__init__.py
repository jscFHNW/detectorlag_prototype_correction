__version__ = '0.1.0'

import os
import numpy as np
import shutil
from subprocess import call, Popen
from PIL import Image
from pathlib import Path
from datetime import datetime
import time
from scipy.ndimage import gaussian_filter

start = time.time()

# Previous image contribution factor
contrib_factor = 0.1
sim_factor = 0.1

# Sigma for gauss filter
sigma = 1
# Directories
ct_dir = f'C:\\Users\\Jonathan Schaffner\\FHNW_Projct\\IP5\\GeneratedData\\SimulatedAfterglowImg\\Scans\\{sim_factor}'
output_base_dir = 'C:\\Users\Jonathan Schaffner\\FHNW_Projct\\IP5\\GeneratedData'

# MuhRec config
muhrec="C:\\Users\\Jonathan Schaffner\\FHNW_Projct\\IP5\\muhrec\\MuhRec.exe"
cfgpath="C:\\Users\\Jonathan Schaffner\\FHNW_Projct\\IP5\\woodRecon.xml"

# Image file name properties
postfix = '####.tif'
prefix_ct = 'merged_'
prefix_dc = 'dc_'
prefix_ob = 'ob_'
prefix_merged='corrected_'
recon_filemask = prefix_merged + postfix
number_fill = postfix.count('#')

# setup directories and filenames
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path(os.path.join(output_base_dir, 'CorrectAfterglowSim', timestamp + '__' + prefix_ct + f'Sim={sim_factor},Contrib={contrib_factor},Sigma={sigma}'))
scans_dir = Path(os.path.join(output_dir, 'Scans'))
recon_dir = Path(os.path.join(output_dir, 'Recon'))

coef_output_dir_raw           = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}xG(U-1,{sigma})"))
coef_output_dir_raw_no_gauss  = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}x(U-1)"))
coef_output_dir_corr          = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}xG((U-1)',{sigma})"))
coef_output_dir_corr_no_gauss = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}x(U-1)'"))
coef_output_dir_raw_no_DC     = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}x(G(U-1,{sigma})-DC)"))
coef_output_dir_raw_no_DC_g   = Path(os.path.join(scans_dir, f"U'=U-{contrib_factor}xG(U-1-DC,{sigma})"))


output_dirs = [coef_output_dir_raw,
               coef_output_dir_raw_no_gauss,
               coef_output_dir_corr,
               coef_output_dir_corr_no_gauss,
               coef_output_dir_raw_no_DC,
               coef_output_dir_raw_no_DC_g]

# Create empty directoris
for sub_dir in output_dirs :
    sub_dir.mkdir(parents=True)

# file name of the images
ct_files = []
ob_files = []
dc_files = []

# images as PIL objects
ct_imgs = {}
ob_imgs = {}
dc_imgs = {}

# images as np arrays
ct_imgs_arr = {}
ob_imgs_arr = {}
dc_imgs_arr = {}
dc_arr = np.array([])

muhrec_instances = []

def main():

    load_images()

    # get average
    dc_avr = np.mean(list(dc_imgs_arr.values()), axis=0)
        
    coef_label = str(round(contrib_factor, 2))

    # set/reset prev_image
    prev_raw_image = np.zeros_like(dc_avr)
    prev_corr_image = np.zeros_like(dc_avr)

    print(f"Processing images with contribution factor {coef_label}!")

    # iterate through projections
    for idx, file_name in enumerate(ct_files) :

        img = ct_imgs[file_name]

        # get tiffinfo from current image
        info = img.tag_v2
        
        # current uncorrected image
        img_arr = ct_imgs_arr[file_name]
        
        # apply correction formula with the uncorrected previous image
        actual_img_arr_raw          = img_arr - contrib_factor * gaussian_filter(prev_raw_image, sigma)        # U' = U - a * G(U-1)
        actual_img_arr_raw_no_gauss = img_arr - contrib_factor * prev_raw_image                                # U' = U - a * (U-1)
        
        # apply correction formula with the corrected previous image
        actual_img_arr_corr          = img_arr - contrib_factor * gaussian_filter(prev_corr_image, sigma)      # U' = U - a * G((U-1)')
        actual_img_arr_corr_no_gauss = img_arr - contrib_factor * prev_corr_image                              # U' = U - a * (U-1)'

         # apply correction formula with the uncorrected previous image and subtracting the DC
         # first image factor is 0
        actual_img_arr_raw_no_DC    = img_arr - (contrib_factor if idx > 0 else 0) * (gaussian_filter(prev_raw_image, sigma) - dc_avr)       # U' = U - a * (G(U-1)-DC)
        actual_img_arr_raw_no_DC_g  = img_arr - (contrib_factor if idx > 0 else 0) * gaussian_filter(prev_raw_image - dc_avr, sigma)        # U' = U - a * G(U-1-DC)

        # create corrected image
        actual_img_raw = Image.fromarray(np.clip(actual_img_arr_raw, 0, 65535).astype('uint16'))
        actual_img_raw_no_gauss = Image.fromarray(np.clip(actual_img_arr_raw_no_gauss, 0, 65535).astype('uint16'))
        actual_img_corr = Image.fromarray(np.clip(actual_img_arr_corr, 0, 65535).astype('uint16'))
        actual_img_raw_corr_no_gauss = Image.fromarray(np.clip(actual_img_arr_corr_no_gauss, 0, 65535).astype('uint16'))
        
        actual_img_raw_no_DC = Image.fromarray(np.clip(actual_img_arr_raw_no_DC, 0, 65535).astype('uint16'))
        actual_img_raw_no_DC_g = Image.fromarray(np.clip(actual_img_arr_raw_no_DC_g, 0, 65535).astype('uint16'))

        # save image with tiffinfo from original image to preserve metadata
        actual_img_raw.save(os.path.join(coef_output_dir_raw , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)
        actual_img_raw_no_gauss.save(os.path.join(coef_output_dir_raw_no_gauss , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)
        actual_img_corr.save(os.path.join(coef_output_dir_corr , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)
        actual_img_raw_corr_no_gauss.save(os.path.join(coef_output_dir_corr_no_gauss , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)
        
        actual_img_raw_no_DC.save(os.path.join(coef_output_dir_raw_no_DC , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)
        actual_img_raw_no_DC_g.save(os.path.join(coef_output_dir_raw_no_DC_g , prefix_merged + str(idx).zfill(number_fill) + ".tif"), format='TIFF', tiffinfo=info)

        # set previous img to current img
        prev_raw_image = img_arr
        prev_corr_image = actual_img_arr_corr

    recon()

    # wait for all instances to be finished
    print("Waitung for reconstruction to finish . . .")
    finished = False
    while(finished == False):
        finished = True
        for p in muhrec_instances:
            if p.poll() == None:
                finished = False
        time.sleep(0.05)

    print("Finished!")
    end = time.time()
    print("Time elapsed: " + str(end - start) + " seconds . . .")


# reconstruct using MuhRec with CLI params
def recon():

    for sub_dir in output_dirs :
        recon_label = os.path.basename(sub_dir)
        print(f"Starting reconstruction for formula {recon_label}")

        recon_label = os.path.basename(sub_dir)

        coef_input_mask = os.path.join(sub_dir, recon_filemask)
        coef_output_dir = os.path.join(recon_dir, recon_label)
        Path(coef_output_dir).mkdir(parents=True)

        # Additional config
        # first_slice=350
        # last_slice=450
        
        # # select projection sub set
        # first_index="projections:firstindex="+str(first_slice)
        # last_index="projections:lastindex="+str(last_slice)

        # set file mask for projections
        file_mask="projections:filemask=" + coef_input_mask

        # recon_slices = "projections:roi=" +

        # set output path for the matrix
        matrix_path="matrix:path=" + coef_output_dir

        # call the reconstruction
        # call([muhrec, "-f", cfgpath, file_mask, matrix_path])
        muhrec_instances.append(Popen([muhrec, "-f", cfgpath, file_mask, matrix_path]))



# loads all images from the output_dir into global variables
def load_images():

    # load all files
    all_files = os.listdir(ct_dir)

    # get OB images
    global ob_files
    ob_files = list(filter(lambda x: x.startswith(prefix_ob), all_files))

    # get DC images
    global dc_files
    dc_files = list(filter(lambda x: x.startswith(prefix_dc), all_files))

    # get projection images
    global ct_files
    ct_files += list(filter(lambda x: x.startswith(prefix_ct), all_files))

    # copy DC images
    for file in dc_files :
        source = os.path.join(ct_dir, file)

        # load images
        dc_imgs[file] = Image.open(source)
        dc_imgs_arr[file] = np.array(dc_imgs[file])

        # Copy to subfolder for each formula
        for formula in output_dirs :
            
            target = os.path.join(formula, file)
            shutil.copyfile(source, target)

    # copy OB images
    for file in ob_files :
        source = os.path.join(ct_dir, file)

        # load images
        ob_imgs[file] = Image.open(source)
        ob_imgs_arr[file] = np.array(ob_imgs[file])

        # Copy to subfolder for each formula
        for formula in output_dirs :
            
            target = os.path.join(formula, file)
            shutil.copyfile(source, target)

    for file in ct_files :
        source = os.path.join(ct_dir, file)

        # load images
        ct_imgs[file] = Image.open(source)
        ct_imgs_arr[file] = np.array(ct_imgs[file])

if __name__ == "__main__":
    main()