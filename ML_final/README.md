# Gender-Classifier-Using-Voice

SVM classifier model saved with 97% test accuracy to 'finalised_model.sav'  and Neural Network classifier model saved to 'neural_network.h5' using <a href="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/">this dataset</a>.
This has been created for distributing and testing on different languages,accents and on different people to compare different models.

### Packages required - (pip install)
 1. librosa
 2. matplotlib
 3. pyaudio
 4. sklearn
 5. pandas
 6. scipy
 7. numpy
 8. wave
 9. keras

### Steps
 1. Download the full <b>"ML_final"</b> folder only, at a specific path (preferred C:\\ )
 2. Open 'svm.py' or 'neural_network.py' and set the path variable to your downloaded folder (path="C:\\\\ML_final\\\\") 
 3. Run the 'svm.py' file and test your results for rbf kernal SVM model or try 'neural_network.py' for neural network model