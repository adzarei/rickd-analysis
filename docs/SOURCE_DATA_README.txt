RUNNING INJURY CLINIC GAIT DATASET
---------------------------------------------------------------

Contained within this dataset are 4 categories of files. They consist of datafiles (.JSON format ->2506 files), metadata (.CSV format ->2 files), Matlab processing code (.M, .MAT format -> 8 files) and Matlab (.M, .MLX, .MAT format -> 8 files) tutorial files.
All code which accompanies this dataset (processing and tutorials) can be found in the "supplementary_materials.zip" file. 

Data files -> 2506 files from 1798 unique subjects
Metadata -> 2 files (1 for walking data, 1 for running data)
Matlab processing code -> 8 files total, 7 of which are necessary to calculate the descriptive variables found in the .JSON data files. The other file consists of a wrapper script which demonstrates running the data through the analysis pipeline.
Tutorial code -> 8 files total consisting of 5 data analysis tutorials and 3 datafiles representing saved outputs from the tutorials which can be loaded by the subsequent tutorial to speed up analysis. Note, the tutorial code also requires some files from the Matlab
	          FileExchange to run. See the tutorial comments for more information.


METHODS
---------------------------------------------------------------

Three-dimensional (3D) marker trajectory data were captured using either a 3-camera or an 8-camera VICON motion capture system (Bonita or MX3+, Vicon Motion Systems Oxford, UK) while participants walked or ran on a 
treadmill. Spherical retro-reflective markers were placed on anatomical landmarks and rigid plates with clusters of 3-4 markers were placed on each of seven lower body segments as per 
Pohl et al. (Gait Posture. 2010;32(4):559-563.).  The marker-set consisted of seven rigid segments and followed International Society of Biomechanics standards. To allow for unobstructed movement during running, 
anatomical markers were removed following a one second static trial where subjects stood upon a template with their feet positioned straight ahead and 0.3m apart with arms crossed over their chest. 
Following a warmup period of 2-5mins, kinematic data were collected for approximately 60 seconds while participants walked and then ran at a self-selected speed.

Data were collected at the University of Calgary Running Injury Clinic as part of research studies or as part of clinical practice between 2009 and 2017. All subjects provided informed consent and all data were
collected under approval from the University of Calgary's Conjoint Health Research Ethics Board (CHREB) (Ethics IDs: E–21705, E–22194, E–24339).


DATA STRUCTURE
---------------------------------------------------------------

Data files are contained within the zipped folder "ric_data" which contains a series of folders with names representing the subject ID. Each subject ID folder contains timestamped datafile(s) in ".json" format with each containing walking and/or running
 data from the single collection session.


.JSON DATA FILES
--------------------------------------------------------------

Raw data is stored in .JSON data format. These files contains the following fields:

hz_w:       Sampling frequency for the walking data 
hz_r:       Sampling frequency for the running data
joints: 	3D coordinates of the markers placed directly on the subject's skin used to mark the subject's joint centres while standing with a neutral posture.These do not represent the actual joint centres as these are calculated from the marker positions.
neutral: 	3D coordinates of the markers within marker clusters strapped to the subject's body while standing with a neutral posture.
walking: 	3D coordinates of the markers on rigid marker clusters strapped to the subject's body while walking on a treadmill.
running: 	3D coordinates of the markers on rigid marker clusters strapped to the subject's body while running on a treadmill.
dv_w:       Descriptive variables calculated from the walking data  (see section -> Matlab Processing Code).
dv_r:       Descriptive variables calculated from the running data (see section -> Matlab Processing Code).


.CSV METADATA
---------------------------------------------------------------

Two files contain the metadata for all .json data files. These files are named: "run_meta_data.csv" and "walk_meta_data.csv". 

These files contain the following column headers:

sub_id: 		Subject identification number
datestring: 	Date and time of data collection
filename: 		Raw data filename.
speed_w(r): 	Calculated walking or running speed in m/s
age: 		Subject age in years at time of collection.
Height: 		Subject height in cm (measured) at time of collection.
Weight: 		Subject weight in kg (measured) at time of collection.
Gender: 		Subject gender
DominantLeg: 	Subject dominant leg
InjDefn: 		Injury severity. Subjects selected from 1 of 4 options: No Injury, Continuing to train in pain, training volume/intensity affected, 2 workouts missed in a row. 
InjJoint: 		Joint location of injury. If not applicable recorded as "No Injury" or empty cell.
InjSide: 		Injury side (right/left).  If not applicable recorded as "No Injury" or empty cell.
SpecInjury: 	Specific injury diagnosis.
InjDuration: 	Injury duration in days.
InjJoint2: 		Secondary injury joint location. If not applicable recorded as "No Injury" or empty cell.
InjSide2: 		Secondary injury side. If not applicable recorded as "No Injury" or empty cell.
SpecInjury2: 	Specific secondary injury diagnosis.
Activities: 	Subject reported athletic activities performed on a regular basis
Level: 		Subject reported level of athletic activity (recreational/competitive)
YrsRunning: 	Number of years subject has been running on a regular basis
RaceDistance: 	Perferred race distance
RaceTimeHrs: 	Perferred race distance best time: Hrs
RaceTimeMins:  	Perferred race distance best time: Mins
RaceTimeSecs: 	Perferred race distance best time: Secs
YrPR: 		Year of preferred race distance personal best time
NumRaces: 		Number of races completed per year.

Note: A subject is considered UNINJURED if "InjDefn" = 'No injury', "InjJoint" = 'No injury' or empty("NULL" / "N/A") and "SpecInjury" is empty("NULL" / "N/A"). "InjJoint" may contain 'No Injury, no injury' which was initially used to indicate there was no secondary injury recorded. 

Note: Some data cleaning should be performed on the injury data in order to produce the best quality query results. To get the same results as Table 1 of the paper accompanying this dataset, please consider the following:

	->  all injuries were converted to only lowercase characters
	-> 'oa', 'hip oa', 'knee oa', and 'osteroarthritis' were all combined into one injury:'osteoarthritis'
	-> 'itb syndrome' and 'itbs' were combined into one injury: 'itb syndrome'
	-> 'patellofemoral pain syndrome' and 'pfps' were combined into one injury: 'patellofemoral pain syndrome'
	-> Important considerations: subjects can have multiple dates when they were tested and could have different injury afflictions each time. 
		-> Subjects who were retested at different times, and were afflicted with the same injury each time, are only counted once in that injury category count.
		-> Subject who were retested at different times, and were afflicted with a different injury each time, are counted once for each applicable injury category count.
		-> Subjects who performed multiple trials on the same day with an injury affliction are only counted once for that specific injury count. 
		-> Averages (physical measurements, and speeds) are based on the first dated session for that subject within the specific injury category 
		   (if a subject came in multiple times over a multi-year span, only the first session was used in the averages calculation).	


MATLAB PROCESSING CODE
---------------------------------------------------------------

All Matlab code was tested using Matlab version 2023a. Matlab code is provided which can be used to calculate the descriptive variables found in the .json data files as well as generate normalized time series curves. 
These files are all commented, please open for more details regarding the functions. A brief description is found below:

gait_kinematics.m		Uses the data contained with the .json raw data files to calculate joint angles and velocities.
gait_steps.m		Uses joint angles and velocities from "gait_kinematics.m" to generate normalized time series curves and calculate descriptive variables. 
gaitClass.m			Classification Descriminant used to determine if a subject is running or walking.
processing_code_example.m	Wrapper script containing an example of processing a .json data file through the Running Injury Clinic pipeline. Calls both "gait_kinematics.m' and "gait_steps.m"
pca_td.m/pca_to.m		Applies a pre-trained algorithm to detect touchdown events.
event_data_TD.mat/event_data_TO.mat		Coefficients of the pre-trained PCA.

Note: It is the responsibility of the user to implement their own data cleaning methods. An example of how this can be done is found in tutorial #2.


MATLAB TUTORIALS
---------------------------------------------------------------

tutorial_1.mlx		Tutorial demonstrating loading the json files, the CSV metadata and a simple analysis of discrete variables to examine fundamental principles of human locomotion.
tutorial_2.mlx		Tutorial demonstrating sample matching and splitting, examination of demographic data, and joint angle calculations/visualizations.
tutorial_3.mlx		Tutorial demonstrating the application of techniques for quantitatively determining differences in joint angles.
tutorial_4.mlx		Tutorial demonstrating outlier removal and unsupervised clustering of the data. 
tutorial_5.mlx		Tutorial demonstrating dynamic metrics of the time series data, rejection sampling of select data subsets, and statistical analysis to compare groups.
tutorial_2.mat		Saved output from "tutorial_2.mlx", which can be input into "tutorial_3.mlx" to save processing time
tutorial_4_cache.mat	Saved output from "tutorial_4.mlx", which can be input into "tutorial_5.mlx" to save processing time
tutorial_5_cache.mat	Saved output from "tutorial_5.mlx", which can be input into future analysis to save processing time

LICENSING
---------------------------------------------------------------

The data is protected under a CC BY 4.0 license.
All scripts and functions are protected under a permissive MIT license which can be found in the file LICENSE.txt. 












