# **Battery-Disassembly**

### **Automated Battery Disassembly using Computer Vision and Task planning.**
- Pretrained YOLOv5 with custom annotated screw data of 4 classes and hosted it on Roboflow.
- Saved the weights as `best_weights.pt` after pretraining on Colab and transferred them to a local machine for USB 3.2 port access.
- Performed screw detection using the pretrained YOLOv5 on static images.
- Calibrated the Intel RealSense D435i camera and used for real-time frame-by-frame inference.


### Some critical challenges overcome
- Built a custom dataset by combining multiple sources on Roboflow due to the absence of a ready-reckoner dataset.
- Utilized Google Colab T4 GPU for ~45 minutes to address the requirement of a large GPU.
- Stored pretrained model weights locally to overcome the challenge of accessing the local USB from Colab.
- Shifted from Python 3.13 to 3.11 as Pyrealsense2, the Python wrapper for Intel Realsense SDK, supports only up to Python 3.11.
- The camera requires USB 3.2 connection to facilitate speedy data tranfer. A lower version will not work as the camera detects depth in-house and sends back info at around ~10 Gbps.



### **Next Steps**

- Introduce depth and relative positioning functionalities.
- Attempt to implement identification of sizes of parts.
- Ensure synergy with hardware for real-time use cases.
- Integrate results with task planning algorithms.





Regards

Arshit




