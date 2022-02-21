"""==============================================================================
 
 Title              :setting_land_cover.py
 Description        :Create the land cover text file necessary for SHETRAN
 Author             :LF Velasquez - I Rohrmueller 
 Date               :Feb 2022
 Version            :1.0
 Usage              :setting_land_cover.py
 Notes              :
                    - Before starting the process the files containing the sys 
                      path and env path for qgis need to be created. Check: ADD LINK TO WEBSITE
python version      :3.8.7
 
=============================================================================="""
# =============================================================================
# Setting all packages
# =============================================================================

from operator import index
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# Global variables
# =============================================================================

# Setting path to work environment
p = Path(__file__)
dir_abs = p.parent.absolute()


# =============================================================================
# Adding all default configurations for QGIS
# =============================================================================

# Setting up system paths
qspath = Path(dir_abs / 'QGIS_env/qgis_sys_paths.csv')
paths = pd.read_csv(qspath).paths.tolist()
sys.path += paths

# Setting up environment variables
qepath = Path(dir_abs / 'QGIS_env/qgis_env.json')
js = json.loads(open(qepath, "r").read())
for k, v in js.items():
    os.environ[k] = v

# In special cases, we might also need to map the PROJ_LIB to handle projections
# for mac OS
# os.environ['PROJ_LIB'] = '/Applications/QGIS-LTR.app/Contents/Resources/proj/'

# QGIS library imports
import PyQt5.QtCore
from PyQt5.QtCore import *
import qgis.PyQt.QtCore
from qgis.core import *
from qgis.analysis import QgsNativeAlgorithms

# Initialising processing module
QgsApplication.setPrefixPath(js["HOME"], True)
qgs = QgsApplication([], False)
qgs.initQgis() # Start processing module

# Import processing
import processing
from processing.core.Processing import Processing
Processing.initialize()

# '''At this point all QGIS libraries and spatial algorithms are available'''


# =============================================================================
# End of default configuration for QGIS
# =============================================================================

# =============================================================================
# Start Process
# =============================================================================

# Step 1. Setting catchment and land cover data ready for work
vlayer_grid = QgsVectorLayer(str(Path(dir_abs / 'Data/outputs/catchm_mask.shp')), 'Catch_layer', 'ogr')
rlayer_LC = QgsRasterLayer(str(Path(dir_abs / 'Data/inputs/LandCover.tif')), 'LC_layer')

# Step 2. Running zonal histogram - QGIS
# Zonal histogram output file
output_ZH = str(Path(dir_abs / 'Data/outputs/ZonalHistogram.csv'))
# Setting and running zonal histogram
zonal_histogram_params = { 'COLUMN_PREFIX' : 'LC_', 'INPUT_RASTER' : rlayer_LC, 'INPUT_VECTOR' : vlayer_grid,
 'OUTPUT' : output_ZH, 'RASTER_BAND' : 1 }
processing.run("qgis:zonalhistogram", zonal_histogram_params)

# Step 3. Working with output of zonal histogram
# Setting col names - this is currently hardcoded as different datasets will use different land cover types
# This will need improving so it either asks for number of land cover types or it reads it from raster file
col_names_lgt = ['id','LC_0','LC_1','LC_2','LC_3','LC_4','LC_5','LC_6','LC_7','LC_9','LC_10','LC_11','LC_12','LC_14','LC_20','LC_21']
col_names_all = ['id','X', 'Y', 'LC_0','LC_1','LC_2','LC_3','LC_4','LC_5','LC_6','LC_7','LC_9','LC_10','LC_11','LC_12','LC_14','LC_20','LC_21']
# Reading csv file and setting dataframe ready for work
df_all = pd.read_csv(output_ZH,usecols=col_names_all)
'''creating dataframe to remove coordinates to avoid errors when finding the LV with the largest coverage'''
df_lgt = df_all[col_names_lgt]

# Step 4. Finding land cover type with the largest coverage per cell
# This gets the largest value per row df_lgt.max(1)
# This finds the position of the value in the dataframe - returns booleans df_lgt.eq(df_lgt.max(1), axis=0).
# This gets the name of the column where the value is True for each row .dot(df_lgt.columns)
df_largest = df_lgt.eq(df_lgt.max(1), axis=0).dot(df_lgt.columns)
# The process returns a series - change to dataframe and add column names
df_largest = df_largest.to_frame().reset_index()
df_largest.columns = ['id', 'LC_largest']
# ID needs to be recalculated to start with 1 to match the original dataframe
df_largest['id'] = np.arange(1, len(df_largest) + 1)

# Step 5. Adding largest land cover type to main dataframe
df_LC = df_all.merge(df_largest, how='right', left_on='id', right_on='id')
# Removing prefix and change value to integer
df_LC['LC_largest'] = df_LC['LC_largest'].str.replace('LC_','').astype(int)
# Replacing 0 with -9999
df_LC.loc[df_LC['LC_largest'] == 0, 'LC_largest'] = -9999

# Step 6. Pivoting dataframe to replicate SHETRAN format
# Pivoting dataframe using X as column and Y as rows 
df_pivot = df_LC.pivot(index='Y', columns='X', values='LC_largest')
df_pivot = df_pivot.sort_index(ascending=False)
# print(df_pivot)

# Step 7. Saving dataframe as text file
filename = Path(dir_abs / 'Data/outputs/final_land_cover_SHETRAN.txt')
np.savetxt(filename, df_pivot.values, fmt='%d')

# Step 8. Creating header needed for SHETRAN
# Defining text file header
ncols = df_pivot.shape[1]
nrows = df_pivot.shape[0]
xllcorner = int(list(df_pivot.columns)[0])
yllcorner  = int(df_pivot.index[-1])
cellsize = 5000
no_data_val = -9999

# Step 9. Adding header to .txt file
# Copying current information in text file
append_copy = open(filename, "r")
original_text = append_copy.read()
append_copy.close()
# Adding header information - this deletes any information in the text file
append_copy = open(filename, "w")
append_copy.write(
    "ncols         " + str(ncols) + "\n" + 
    "nrows         " + str(nrows) +  "\n" +
    "xllcorner     " + str(xllcorner) +  "\n" +
    "yllcorner     " + str(yllcorner) + "\n" +
    "cellsize      " + str(cellsize) + "\n" +
    "NODATA_value  " + str(no_data_val) + "\n")
# Pasting content that was in .txt file before the header
append_copy.write(original_text)
# Saving .txt file
append_copy.close()


# =============================================================================
# End Process
# =============================================================================

print('-----')
print('The process has ended!! Go and check.')
print('-----')


# =============================================================================
# Exit the QGIS processing module
# =============================================================================
qgs.exitQgis()