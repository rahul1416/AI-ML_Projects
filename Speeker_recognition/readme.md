There are two different method to store the features of audio file one is mfcc(https://link.springer.com/content/pdf/bbm:978-3-319-49220-9/1.pdf)

mfcc is used for feature extraction of an audio file so i have taken 100 features of audio and store them to in a csv 
the main reason to do the mfcc is that there are approx 10000 audio files in the main dataset 
which is given the spuc (https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) websit and it has properly defined the speaker(M01/male).

also major reason to do that it will not require CNN can be classified by using simple ANN and DescitionTreeClassifier

In Assignment2_spectrogram used the dataset (https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets)
there are approx 800 audio file and used CNN(in pytorch) for classification 

