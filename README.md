# Video_Processing_using_GPU

### I have implemented the video capturing which uses GPU to process it using CUDA in Google Colab.

**Step 1:** Google Colab is a free service with powerful NVIDIA GPU. To set up the Colab environment for OpenCV with GPU we need a few lines of code.

1.1: Clone the latest OpenCV build from GitHub.
```python
    #Clone the OpenCV repository
    !git clone https://github.com/opencv/opencv
    !git clone https://github.com/opencv/opencv_contrib

    #Create a build folder and navigate to it  
    !mkdir build
    %cd build
```
1.2: Next, run the cmake command after setting the parameters WITH_CUDA=ON and finally, run the make command below.
```python
    !cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF 
    -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D WITH_OPENEXR=OFF -D WITH_CUDA=ON -D WITH_CUBLAS=ON 
    -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON ../opencv
```
**Step 2:** To keep data in GPU memory, OpenCV introduces a new class  `cv2.cuda_GpuMat` in Python which serves as a primary data container. 
It helps in making the transition to the GPU module as smooth as possible.

**Step 3:** First, we have to load the video on the CPU before passing it (frame-by-frame) to GPU. `cv.VideoCapture()` can be used to load and iterate through video frames on the CPU.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
```
**Step 4:** Now to create a frame on GPU for images, we use `cv.cuda_GpuMat()`.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
    
    gpu_frame = cv2.cuda_GpuMat()
```
**Step 5:** For each frame to be uploaded on GPU, we use `frame.upload()`, which copies data from host memory to device memory.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
    gpu_frame = cv2.cuda_GpuMat()
  
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()

        gpu_frame.upload(frame)
```
**Step 6:** To see the images after processing, we need to bring each back from GPU memory (cv2.cuda_GpuMat) to CPU memory (numpy.ndarray).
To do this we use `frame.download()`.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
    gpu_frame = cv2.cuda_GpuMat()
  
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()

        gpu_frame.upload(frame)
        frame = cv2.cuda.resize(gpu_frame, (852, 480))
        frame.download()
```
Finally below is the code for video capturing process which uses a GPU to process it.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
    gpu_frame = cv2.cuda_GpuMat()
  
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()

        gpu_frame.upload(frame)
        frame = cv2.cuda.resize(gpu_frame, (852, 480))
        frame.download()
        
        # Display the resulting frame
        cv2_imshow(frame)
      
        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
```
Below was the code for video capturing process which uses a CPU to process it.
```python
    import cv2
    from google.colab.patches import cv2_imshow
    # define a video capture object
    vid = cv2.VideoCapture('/content/Pexels Videos 1572442.mp4')
  
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()
        frame = cv2.resize(frame, (640, 360))
  
        # Display the resulting frame
        cv2_imshow(frame)
      
        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
```
