{\rtf1\ansi\ansicpg1252\cocoartf1404\cocoasubrtf470
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \



Train-Test-data/ contain two matlab files converting .png files into one csv files
for training and testing set. Please download the Format1 file from the website:

http://ufldl.stanford.edu/housenumbers/


Tensorflow-cnn-SVHN-4-4layers-Dropout-40/\
\
contains convolution neural network of 4 layers and drop out applies to each layer but\
not the input layer. The pictures are of three channels and 40 by 40 pixels.\
\
Tensorflow-cnn-SVHN-4layers-DropoutMaxout54grayscale/\
\
contains convolution neural network of 4 layers and drop out applies to each layer but not the \
input layer. The picture are  of 54 by 54 pixels.\
\
The svhn.py file will do the training and testing for you on training and testing set, after
training, the model.ckpt file will be generated.
\
Online-testing/ help with the real short number sequence recogintion. 

Shape the png file to 40 times 40, 3 channels and copy it into the same folder. 
Open the RestorePNG.py.

In the main function, such as 
def main():
img=mpimg.imread('28.png');

Put the name of the png file that you want to do the recognition. Run the RestorePNG.py.


}# Project-OCR
