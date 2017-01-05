The implementation was used in:

[1] Ertunc Erdil, Fitsum Mesadi, Tolga Tasdizen, Mujdat Cetin, 
“Disjunctive Normal Shape Boltzmann Machine”, 
IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans.

Any papers using this code should cite [1] accordingly.

The software has been tested under Matlab R2015a.

After unpacking the file and installing the required libraries,
start Matlab and run testDNSBM.m to run the DNSBM on the test images in the paper.
You can select which test image to use by changing file names in testDNSBM.m. These lines
are commented in the code. You do not need to train DNSBM from scratch, the parameters
of DNSBM are already in the parameters.mat files in the corresponding data set folder.

If you would like to train DNSBM for your own application, please construct training images in the format given
in the data set folders. Then, you can train your own DNSBM using trainDNSBM.m

If you have problems, you can email me at ertuncerdil@sabanciuniv.edu
I will do my best to help.

Please also report any bug to ertuncerdil@sabanciuniv.edu
