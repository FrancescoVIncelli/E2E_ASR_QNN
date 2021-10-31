# End-to-End Automatic Speech Recognition using Quaternion Convolutional Neural Networks
The repository contains two Google Colaboratory notebooks implementing the project respectively in the TensorFlow and PyTorch deep learning frameworks.
Further theoretical background informations and technical details are provided in detail in the *Report* section in the [MLSP]Project_Vincelli_(tensorflow).ipynb Colaboratoy notebook.

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
├── Report                   - Brief introduction of theory, motivations and proposed solution
├── Dependencies             - Installation of required packages
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
- residual time-distributed network architecture
- temporal convolutional architecture  

Some changes and improvements of the network architectures are also experimented, defined different versions of the same architectures.  
- The PyTorch notebook in particular shows an improvement of the ResNet architecture by employing real/quaternion-valued recurrent layers in the classification block on top of the network.  Moreover the LibriSpeech dataset is loaded using the torchaudio package utilities.
- The TensorFlow notebook provides with a preliminary script that creates a .jsonl file for the training and the evaluation/test sets used in the features extraction procedure.  This process could require some time and for this reason the trainijng and validation .jsonl files are already provided in this repository.  

Each of the notebooks also contains a preliminary section with a full and detailed analysis and visualization of the data used for the experimentations.  

### Execution of the code
- The only file that is to be run is the [main.m](code/main.m) file. Make sure to select the current path to be [code](code/) before running the script
- The [main.m](code/main.m) file contains "Initialize Parameters" section where the frequencies and densities are to be set 
- To get the plots, individual scripts have to be run to get degrees, densities and efficiency plots. Each of them has comparision functions in which individual samples can be comparte with the others. Make sure to uncomment the necessary plot before running the script. 

### Limitations
- The code is code to work for 21 channels and 64 channels. The indices should be updated if the number of channels are different
- The project computes only the general kind of PDC (Baccalà and Sameshima, 2001) and one kind of normalization technique (Astolfi et al, 2007). Other PDC computation techniques can be implemented to get better results
- All the plots are coded to work for only this application will have to be slightly modified based on the requirements. The functions are general which can be used for a differnt application with slight modification 
