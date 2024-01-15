you need to download the dataset SPEECH_DATA_ZIPPED from the link (https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip) since the dataset is very large approx 4 GB 

unzip speech data SPEECH_DATA_ZIPPED and the folder in folder 
	'SPEECH_DATA_ZIPPED/SPEECH DATA/MALE/.../.wav'

run the Data_storage.ipynb file for storing the mfcc.csv in which there are extracting 100 features of mfcc(Mel-frequency cepstral coefficients) 

for running 'Classification_Task2.ipynb' you don't require to download the SPEECH_DATA_ZIPPED dataset because i have stored the feaures of audio in mfcc.csv file

After the storing of data then run 'Classification_Task2.ipynb'

The model are saved in the model folder for classification of speekar the are 40 classification like 10 male speacker and 10 female speaker also two folder LAR(with some random Noise) and MIC(without Noise) so the classification as follows -> Female-Mic-M01 

in 'Classification_task2.ipynb' in last i have added some noise in check the model that it can classify the model accuratly but there is no such chance that i will be able to define speacker with noise accuratly.
