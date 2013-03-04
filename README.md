Cloth folding and manipulating
==================================
We are trying to create a system that would be able to master a process of cloth unfolding and folding by two armed robotic manipulator. This system has ambition to create a basis for testing of an another aproaches. For example for to test another cloth models. In this implementation we are using the cloth model developed by by Stephen Miller and his colleagues from Berkley university.

Whole repository is written as a single ROS package. The main parts of the system is written in python script language.

Important files
-----------------------------------
* scripts/FoldingProcess.py - A script that automatize folding process and operation with cloth model.
* scripts/ImageReaderService.py - A script that subscribe images from Kinect and offeres it. It works as an server node.
* srv/GetImage.srv - Service message for a service node that reads images from a Kinect and keep last image in memory.
* manifest.xml - This file contains ros packages dependencies that are used in scripts.

How to make it run
-----------------------------------
All demand packages should be almost found on the git repositories at https://github.com/rll.
Then update all scrips from folder ./modified\_components/
And it should be all.


Instalation step-by-step notes
-----------------------------------
1) Download clothFolding from repository
    rosws set contour_model_folding --git https://github.com/sindljan/clothFolding-berk.git

2) Download depended ros packages
    rosws set visual_feedback --git https://github.com/rll/visual_feedback.git 
    rosws set berkley_utils --git https://github.com/rll/berkeley_utils.git
    rosws set folding --git https://github.com/rll/folding.git

3) Update all packages that were added
    rosws update

4) Make package
    rosmake clopema_cloth_folding

5) Update modified scripts according to description from ./modified\_components/readme file.



Notes
-----------------------------------
I am using this scrips on virtual machine with Ubuntu Precise 12.04 hosted on Win7 32b. It's litle bit trycky to make Kinect works on virtual machine. OpenNi drivers doesn't work. The freenect drivers are the right once. 
* Be careful about Kinect model selection. Model with s.n. 1473 doesn't work for me. It has some problem with freenect libraries and I think also with windows drivers. The Windows drivers are bigger problem then the Linux one.
