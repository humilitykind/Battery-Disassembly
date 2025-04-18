import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream depth and color images
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for frames and get both depth and color frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue  # Skip iteration if no frame is available

        # Convert images to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply color map to depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack color and depth images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow("Intel RealSense D435i", images)

        # Get depth value at centre pixel
        depth = depth_frame.get_distance(320, 240)
        print(f"Depth at centre pixel: {depth:.2f} m")

        





        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()