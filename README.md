# Battery-Disassembly

Automated Battery Disassembly using Computer Vision and Task planning.

We first Pretrained yolov5 with custom annotated screw data of 4 classes and hosted it on roboflow.

As pretraining was done on colab file, we saved the weights as best_weights.pt and transfered it to local machine for having access to USB 3.2 ports.

Then, screw detection was done using a pretrained yolov5 on static images.

Next, we caliberated the Intel RealSense D435i camera and produced frame-by-frame inference.


Next Steps:
Introducing depth, and relative positioning functionalities.

Having synergy with Hardware , for real time use-cases.

Integrating results with task planning algorithms.






Regards
Arshit




