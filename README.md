# deep-mil-for-whole-mammogram-classification

The code is based on keras with theano backend.

Please cite our paper 

Zhu, Wentao, Qi Lou, Yeeleng Scott Vang, and Xiaohui Xie. "Deep Multi-instance Networks with Sparse Label Assignment for Whole Mammogram Classification." MICCAI 2017.

Preprocessed InBreast dataset can be downloaded from https://drive.google.com/drive/folders/0B-7-8LLwONIZZm1pQWdyak5Od28?usp=sharing or https://pan.baidu.com/s/1eMep77riQrXqg9oU5uPAHw. If you use the preprocessed dataset, you do not need to download the original InBreast dataset.

For label processing, please read inbreast.py readlabel() function. For converting dicom to pickle file, please read readdicom() function in the inbreast.py.

For original InBreast dataset (.dicom files), please contact http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database . 

run_cnn_k_new.py is used for alex net.
run_cnn_k_mil_new.py is used for max pooling based deep mil.
run_cnn_k_mysparsemil_new.py is used for sparse deep mil.
run_cnn_k_mymil_new.py is used for label assignment based deep mil. Here we finetuned weights from max pooling based deep mil.

_test is used for test. 

Our code is based on conv-keras and keras. If you use current version of keras, you need to revised the code a little bit for adaption. I will update a repo. for current keras.

If you have any questions, please contact with me wentaozhu1991@gmail.com.
