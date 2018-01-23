Here you can find the implementation of Vincent Matthys, Pirashanth Ratnamogan and Othmane Sayem
for the second project of the Unsupervised Learning class.

The implementation has been done in python 3.
To compute the code you only need some basic librairies:
scipy, numpy and sklearn (for KMeans).
###########################################################################
The algorithm implementations of question 1 are in the root directory.

###########################################################################
You can find the data online :
- http://www.vision.jhu.edu/gpca/fetchcode.php?id=210 for ExtendedYaleB
- http://www.vision.jhu.edu/data/fetchdata.php?id=1 for Hopkins155
Please download them in a data/ directory
###########################################################################
To launch the tests that we did for the question 2 one have to look the file:
run_test_face_clustering.py
###########################################################################
To launch the tests that we did for the question 3 one have to look the notebook:
Motion_segmentation.ipynb

Raw binary results for question 3are in:
motion_res_SC.npy
motion_res_SSC.npy
motion_res_ksub.npy

Run the ipynb Motion_segmentation.ipynb.
You can load the binary results produced by the algorithms by executing the
corresponding cells np.load(motion_res_X.npy).item()


###########################################################################
We tried to do some extra in the folder Extra-implementations-try
However the SSC with corrupted entries doesn't converge properly.
###########################################################################
You can also find the report that summarize our thinking during the project.

Best Regards,

Vincent Matthys, Pirashanth Ratnamogan and Othmane Sayem
