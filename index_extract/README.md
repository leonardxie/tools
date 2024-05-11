1. CRF模型训练 \
运行train.py


2. CRF模型调用 \
测试文件放入corpus路径下，可以为txt文件或者与训练数据一样的格式的数据，运行test.py


3. 训练测试一体化使用 \
运行main.py


4. 追加或创建新的训练数据 \
a. txt文件，命名为data_add.txt，放入corpus文件夹下 \
b. 将txt文件，转化为适合打标签的文件，运行data_tag/write_data_to_xlsx.py \
c. 将打好标签的文件，按照指定格式要求转化为项目可读取的数据文件，运行data_tag/write_taged_data_to_txt.py \
d. 将写好的数据，添加到原始train文件中，data_process.py下的add_data_to_train_bmes方法
e. 将全新的数据，替代原有的train文件，data_process.py下的create_data_to_bmes方法

