git clone https://github.com/jiegenghua/Traffic-Sign-Objection.git
mkdir data
wget http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip
wget http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/code.zip
unzip data.zip
unzip code.zip
cd Mycode
python dataextract.py
python datapreprecessing
python train_test.py
