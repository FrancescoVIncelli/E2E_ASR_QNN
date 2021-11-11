# End-to-End Automatic Speech Recognition using Quaternion Convolutional Neural Networks
The repository contains two Google Colaboratory notebooks implementing the project respectively in the TensorFlow and PyTorch deep learning frameworks.
Further theoretical background informations and technical details are provided in the *Report* section in the [MLSP]Project_Vincelli_(tensorflow).ipynb Colaboratoy notebook.

### Dataset
The experiments described below are performed on the LibriSpeech ASR Corpus. The LibriSpeech corpus is a collection of approximately 1,000 hours of audiobooks that are a part of the LibriVox project. Most of the audiobooks come from the Project Gutenberg. The training data is split into 3 partitions of 100hr, 360hr, and 500hr sets while the dev and test data are split into the ’clean’ and ’other’ categories, respectively.
Below the data archives used for the training and testing of the asr models are reported:  
[train-clean-100.tar.gz](https://www.openslr.org/resources/12/train-clean-100.tar.gz) (training set of 100 hours "clean" speech )  
[dev-clean.tar.gz](https://www.openslr.org/resources/12/dev-clean.tar.gz) (development set, "clean" speech )  
[test-clean.tar.gz](https://www.openslr.org/resources/12/test-clean.tar.gz) (test set, "clean" speech )  


### Installation Procedure

### File Structure
The two notebook files containing the entire project code show the following structure (some differences related to the deep learning framework used are highlithed)

```txt
├── Report
├── Dependencies
├── Implementation
│   ├── Data preprocessing      
│   ├── Data generator and visualization          
│   ├── Training and testing utility function            
│   ├── Data analysis and exploration            
│   └── Acoustic features for ASR             
│
├── Models
│   ==== [TensorFlow] ====
│   ├── Temporal convolutional model                  
│   │   ├── TCN + MLP classifier (Real/Quaternion-valued)                               
│   │   └── TCN 'pure' convolutional model (Real/Quaternion-valued)                              
│   └── Residual Convolutional-TimeDistributed model (Real/Quaternion-valued)               
│
│   ==== [PyTorch] ====
│   ├── Temporal convolutional model                  
│   ├── Residual convolutional model     
│   │   ├── Real-valued 1D/2D Convolutional model
│   │   ├── Quaternion-valued Conv-1D (Orkis library)
│   │   └── Quaternion-valued Conv-1D (Speechbrain library)
│
├── Experimental evaluation
│   ├── ResNet-Real-valued                
│   │   ├── Training setup #1                                
│   │   └── Training setup #2       
│   ├── ResNet-Quaternion-valued                
│   │   ├── Training setup #1                                 
│   │   └── Training setup #2  
│   │   
│   ├── TCN-Real-valued
│   │   ├── Training setup #1                              
│   │   └── Training setup #2  
│   ├── ResNet-Quaternion-valued                
│   │   ├── Training setup #1                                 
│   │   └── Training setup #2  
```

### Documentation
The project implements real/quaternion-valued convolutional neural network models for the task of end-to-end automatic speech recognition on the LibriSpeech ASR corpus dataset.  

Both the two notebooks provide with two model architectures respectively implemented using the TensorFlow and the PyTorch frameworks:
- residual time-distributed convolutional network architecture
- temporal convolutional architecture  

Some improvements of the network architectures are also experimented, defining different versions of the same architectures.  
The PyTorch notebook in particular shows an improvement of the ResNet architecture by employing real/quaternion-valued recurrent layers in the classification block on top of the network.  Moreover the LibriSpeech dataset is loaded and processed using the torchaudio package utilities.  
The TensorFlow notebook provides with a preliminary script that creates a .jsonl file for the training and the evaluation/test sets used in the features extraction procedure.  This process could require some time and for this reason the training and validation .jsonl files are already provided in this repository.  

Each of the notebooks also contains a preliminary section with a full and detailed analysis and visualization of the data used for the experimentations.  

### Execution of the code
In this section the instructions to run the TensorFlow notebook are reported. This notebook has been used to train the models whose training weights are stored in the project folder [results](results).

**On Google Colaboratory**  

1. Create a folder `LibriSpeech` including a Train, Valid and Test subfolders.
2. Extract the data from the archives downloaded fromt the link provided in section [Dataset](#dataset) and put the train, validation and test dataset into the corrsponding subfolders of LibriSpeech directory.
3. Upload the LibriSpeech dataset folder created in *step 1* into your Google Drive.
4. Mount the Gogole Drive filesystem followig the procedure used in the notebook: change the path to the dataset folder if different names for the directories are used.
5. Run the script to create the .jsonl files for train, validation and test set: be aware to change the name of the desired data set (train, validation, test) before to run the script.
6. Run the other sections of the notebook to train the models and evaluate the results for each experiments.

**On Jupiter**  

1. Download or clone this repository in your system.
2. Create a folder `LibriSpeech` including a Train, Valid and Test subfolders in the project directory.
3. Extract the data from the archives downloaded fromt the link provided in section [Dataset](#dataset) and put the train, validation and test dataset into the corrsponding subfolders of LibriSpeech directory.
5. Run the script to create the .jsonl files for train, validation and test set: be aware to change the name of the desired data set (train, validation, test) before to run the script.
6. Run the other sections of the notebook to train the models and evaluate the results for each experiments.
