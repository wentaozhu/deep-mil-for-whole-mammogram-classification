# deep-mil-for-whole-mammogram-classification

Please cite our paper 

Zhu, Wentao, Qi Lou, Yeeleng Scott Vang, and Xiaohui Xie. "Deep Multi-instance Networks with Sparse Label Assignment for Whole Mammogram Classification." MICCAI 2017.

Preprocessed InBreast dataset can be downloaded from https://drive.google.com/drive/folders/0B-7-8LLwONIZZm1pQWdyak5Od28?usp=sharing

For original InBreast dataset (.dicom files), please contact http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database

run_cnn_k_new.py is used for alex net.
run_cnn_k_mil_new.py is used for max pooling based deep mil.
run_cnn_k_mysparsemil_new.py is used for sparse deep mil.
run_cnn_k_mymil_new.py is used for label assignment based deep mil. Here we finetuned weights from max pooling based deep mil.

_test is used for test. 

Our code is based on conv-keras and keras.

If you have any questions, please contact with me wentaozhu1991@gmail.com.
