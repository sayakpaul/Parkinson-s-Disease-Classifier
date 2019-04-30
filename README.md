# Parkinson's Disease Classifier

[Adrian Rosebrock](https://www.linkedin.com/in/adrian-rosebrock-59b8732a) of [PyImageSearch](https://pyimagesearch.com/) recently released a brand new tutorial: [Detecting Parkinson’s Disease with OpenCV, Computer Vision, and the Spiral/Wave Test](https://www.pyimagesearch.com/2019/04/29/detecting-parkinsons-disease-with-opencv-computer-vision-and-the-spiral-wave-test/) 
which shows how to automatically detect Parkinson’s disease in hand-drawn images of spirals and waves. Adrian used classical 
computer vision techniques like _Histogram of Oriented Gradients (HOG)_ for quantifying the features of the images and 
used them to train a _Random Forest Classifier_. He got an accuracy of **83.33%**. 

I decided to apply deep learning to this problem and see if I can push the score. To see if I was able to do this, I would request you to take a look at the accompanying notebook [here](https://github.com/sayakpaul/Parkinson-s-Disease-Classifier/blob/master/Parkinson_s_Disease_Classifier.ipynb).

**Note** that, the data was provided along with the PyImageSearch tutorial mentioned above. In order to make the folder structure more convenient for myself, I arranged it in the following way: 

![](https://i.ibb.co/VCSZtyx/Screen-Shot-2019-04-30-at-12-30-42-PM.png)

Whereas, Adrian's arrangment was a bit different: 

![](https://i.ibb.co/sHj17qt/Screen-Shot-2019-04-30-at-5-22-30-PM.png)

Useful links:
- [PyImageSearch Gurus](https://www.pyimagesearch.com/pyimagesearch-gurus/)
- [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html)
