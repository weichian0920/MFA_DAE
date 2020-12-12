### Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder
<p align="center">
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao1-3016831-large.gif">
</p>
<p align="center">
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6221020/9248684/9167389/tsao4-3016831-large.gif">
</p>

### Abstract
Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder
Auscultation is the most efficient way to diagnose cardiovascular and respiratory diseases. To reach accurate diagnoses, a device must be able to recognize heart and lung sounds from various clinical situations. However, the recorded chest sounds are mixed by heart and lung sounds. Thus, effectively separating these two sounds is critical in the pre-processing stage. Recent advances in machine learning have progressed on monaural source separations, but most of the well-known techniques require paired mixed sounds and individual pure sounds for model training. As the preparation of pure heart and lung sounds is difficult, special designs must be considered to derive effective heart and lung sound separation techniques. In this study, we proposed a novel periodicity-coded deep auto-encoder (PC-DAE) approach to separate mixed heart-lung sounds in an unsupervised manner via the assumption of different periodicities between heart rate and respiration rate. The PC-DAE benefits from deep-learning-based models by extracting representative features and considers the periodicity of heart and lung sounds to carry out the separation. We evaluated PC-DAE on two datasets. The first one includes sounds from the Student Auscultation Manikin (SAM), and the second is prepared by recording chest sounds in real-world conditions. Experimental results indicate that PC-DAE outperforms several well-known separation works in terms of standardized evaluation metrics. Moreover, waveforms and spectrograms demonstrate the effectiveness of PC-DAE compared to existing approaches. It is also confirmed that by using the proposed PC-DAE as a pre-processing stage, the heart sound recognition accuracies can be notably boosted. The experimental results confirmed the effectiveness of PC-DAE and its potential to be used in clinical applications.
### Requirements
* librosa             0.6.2. 
* torch               1.5.1
* torchvision         0.6.0a0+35d732a
* sklearn             0.0
* numpy               1.16.1
* scipy               1.2.1
### Dataset

### How to run
Scripts to reproduce the training and evaluation procedures discussed in the paper are located on scripts/.
