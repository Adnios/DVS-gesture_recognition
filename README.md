# IITM DVS128 Gesture Dataset
---
## About the dataset
This dataset was collected by the members of [Computational Imaging Lab][cilab_link] of the Indian Institute of Technology Madras as a part of the work,
_"Dynamic Vision Sensors for Human Activity Recognition" - Stefanie Anna Baby, Bimal Vinod, Chaitanya Chinni, Kaushik Mitra_, accepted at _the 4th IAPR Asian Conference on Pattern Recognition (ACPR) 2017_.

This dataset is released under the Creative Commons Attribution 4.0 license.

This dataset is available at https://www.dropbox.com/sh/ppjgszi51no884f/AACb3xMxmftHD3P3XqpahIJOa?dl=0

Required disk space: 615.1 MB for the `zip` file and 1.02 GB for the extracted data.

## Contents of the dataset
The **IITM DVS128 Gesture Dataset** contains 10 hand gestures by 12 subjects with 10 sets each totalling to a 1200 hand gestures.
These gestures are captured using a [DVS128 camera][dvs128_link].

The 10 gestures can be listed as follows,
 1. come here
 2. left swipe
 3. right swipe
 4. rotation counter-clockwise
 5. rotation clockwise
 6. swipe down
 7. swipe up
 8. swipe V
 9. symbol X
 10. symbol Z

This dataset contains a folder for each one of the above 10 gestures (total 10 folders).
Each folder contains the gestures of all the 12 subjects with 10 sets each (total 120 files per folder) in `aedat` 2.0 format.
The specifications of AEDAT 2.0 can be found [here][aedat_2.0].

The corresponding dataset in `avi` format is available at [IITM_DVS_10][IITM_DVS_10_link].

## Citation
S. A. Baby and B. Vinod and C. Chinni and K. Mitra, "Dynamic Vision Sensors for Human Activity Recognition,"
2017 4th IAPR Asian Conference on Pattern Recognition (ACPR), Nanjing, China, 2017.


    @inproceedings{sababy:hardvs:2017,
        author={S. A. {Baby} and B. {Vinod} and C. {Chinni} and K. {Mitra}},
        booktitle={2017 4th IAPR Asian Conference on Pattern Recognition (ACPR)},
        title={Dynamic Vision Sensors for Human Activity Recognition},
        year={2017},
        pages={316-321},
        doi={10.1109/ACPR.2017.136},
        ISSN={2327-0985},
        month={Nov}
    }

[cilab_link]: http://www.ee.iitm.ac.in/comp_photolab/
[dvs128_link]: https://inivation.com/support/hardware/dvs128/
[aedat_2.0]: https://inivation.com/support/software/fileformat/#aedat-20
[IITM_DVS_10_link]: https://www.dropbox.com/sh/ppjgszi51no884f/AACb3xMxmftHD3P3XqpahIJOa?dl=0
