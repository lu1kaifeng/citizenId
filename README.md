
# Implementation of "Chinese Citizen ID Card Identification using CNN-RNN-CTC"

<p float="left">
<img  src="https://github.com/lu1kaifeng/citizenId/blob/master/Picture1.png" width="45%">
<img  src="https://github.com/lu1kaifeng/citizenId/blob/master/Picture2.png" width="45%">
</p>

To read my thesis, [click here](https://github.com/lu1kaifeng/citizenId/blob/master/thesis.pdf).
> Various documents play an indispensable role in daily life, such as ID cards, student ID cards, etc., and corresponding documents are needed to be shown when handling business. With the development of Internet technology, many companies have developed their own APP products for their users in order to build their own brands, most of which need to carry out real-name authentication. However, the identification mechanism of many APPs is not perfect, and the identification information is often read incorrectly, resulting in unsatisfactory user experience. In order to make real-name authentication more convenient for end users and respond to the national real-name system policy, this thesis proposes an ID card OCR system based on CNN-RNN-CTC.

>The system uses the SIFT feature matching algorithm to complete the text area matching and correction in a single step. The acquired text area can be directly used as the input of CNN-RNN-CTC, which has the characteristics of high accuracy and fast speed. Through the end-to-end text recognition based on CNN-RNN-CTC, the recognition process does not need explicit text segmentation, but converts text recognition into a sequence learning problem. For input images with different scales and different text lengths, through the CNN and RNN layers, the entire text image can be recognized, and the text cutting can be directly integrated into deep learning, achieving better accuracy in the recognition process.
