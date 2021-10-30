# End-to-End Automatic Speech Recognition using Quaternion Convolutional Neural Networks
The repository contains two Google Colaboratory notebooks implementing the project respectively in the TensorFlow and PyTorch deep learning frameworks.
Further theoretical background informations and technical details are provided in detail in the *Report* section in the [MLSP]Project_Vincelli_(tensorflow).ipynb Colaboratoy notebook.

### Dataset


### Installation Procedure

### File Structure
```txt
hand_networks/
├── README.md                   - Installation and execution procedures, description of code modules and ouput
├── data/
│   ├── nodePos.mat             - Contains the pixel positions of the electrodes used to plot graph 
│   ├── locations.mat           - Contains the variables for measurement systems of 64 channel data format
│   ├── S001R03.edf             - Contains dataset of open and close left or right fist (LHM, RHM)
│   ├── S001R04.edf             - Contains dataset of imagine opening and closing left or right fist (LHI, RHI)
│   ├── S001R07.edf             - Contains dataset of open and close left or right fist (LHM, RHM)
│   ├── S001R08.edf             - Contains dataset of imagine opening and closing left or right fist (LHI, RHI)
│   ├── S001R11.edf             - Contains dataset of open and close left or right fist (LHM, RHM)
│   └── S001R12.edf             - Contains dataset of imagine opening and closing left or right fist (LHI, RHI)
│
├── code/
│   ├── main.m                  - Main file which is to be run to compute the entire routine
```

### Documentation
This project perfroms a sequence of steps as shown below
1. Three parameters `(Frequency1, Frequency2 and density)` are set
2. The dataset is loaded using [dataset_load.m](code/dataset_load.m) script
3. The pixel positions for plotting the graph is loaded which is stored as a `.mat` file using [get_node_pose.m](code/get_node_pose.m) script 
4. The `run_routine` function is called multiple times by changing the dataset, frequecny and number of channels which performs a series of steps
   1. Computes Multi-Variate Autoregressive model and then Partial Directed Coherence to obtain a 3D matrics. One frequency is choosen from the 3D matrics and converted to binary matrix by setting a threshold to obtain given density
   2. Computes the local indices (in-degree, out-degree, degree, left and right hemisphere density)
   3. Computes the global indices (global efficiency and local efficiency)
   4. Computes the graph (edges and nodes) for plotting the graph
   5. Stores all the obtained results into a data structure
5. Finally, all the results are plotted by running their respective plot scripts

### Execution of the code
- The only file that is to be run is the [main.m](code/main.m) file. Make sure to select the current path to be [code](code/) before running the script
- The [main.m](code/main.m) file contains "Initialize Parameters" section where the frequencies and densities are to be set 
- To get the plots, individual scripts have to be run to get degrees, densities and efficiency plots. Each of them has comparision functions in which individual samples can be comparte with the others. Make sure to uncomment the necessary plot before running the script. 

### Limitations
- The code is code to work for 21 channels and 64 channels. The indices should be updated if the number of channels are different
- The project computes only the general kind of PDC (Baccalà and Sameshima, 2001) and one kind of normalization technique (Astolfi et al, 2007). Other PDC computation techniques can be implemented to get better results
- All the plots are coded to work for only this application will have to be slightly modified based on the requirements. The functions are general which can be used for a differnt application with slight modification 
