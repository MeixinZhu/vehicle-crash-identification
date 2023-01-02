from generate_train_data import generate_train
from predict_test_a import predict_a
from generate_train_data_model2 import generate_train_model2
from model1 import predict_model1
from model2 import predict_model2
from combine import combine
from generate_train_data_model_des import generate_train_des
from predict_test_a_des import predict_a_des

if __name__ == '__main__':
    generate_train()
    generate_train_des()
    predict_a_des()
    predict_a()
    generate_train_model2()
    predict_model1()
    predict_model2()
    combine()