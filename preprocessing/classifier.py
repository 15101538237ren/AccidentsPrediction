# -*- coding: utf-8 -*-
from time import time
import keras,math,sys
import numpy as np
from scipy import interp
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Model,load_model,Sequential
from itertools import cycle
from keras.layers import LSTM, Dense, Dropout, Input, GRU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,TensorBoard
from keras.optimizers import RMSprop
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from util import generate_arrays_of_validation,generate_arrays_of_train,generate_arrays_of_test,generate_arrays_of_train_aux,generate_arrays_of_validation_aux
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import LinearSVC,NuSVC,SVR
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from plot_roc import plot_roc
from keras.optimizers import Adadelta
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error,precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
reload(sys)
sys.setdefaultencoding('utf8')
itt_range = 3

def mean_relative_error(real, pred):
    error = float(np.sum((real - pred) / real))/float(real.shape[0])
    return error

#LSTM
def train_and_test_model_with_lstm(aux_input_dim,aux_input_data,data_dim,n_time_steps, save_path,train_data, validation_data, test_data, train_label, validation_label, test_label, class_weight = {0:1,1:1}):
    epochs = itt_range
    epoch_now = 1
    lstm_dim = n_time_steps
    batch_size = 8
    steps_per_epoch = 1#int(math.ceil(float(len(train_label))/float(batch_size)))
    validation_steps = int(math.ceil(float(len(validation_label))/float(batch_size)))
    model_path = save_path+"ckpt_lstm_"+str(epoch_now)+".h5"
    # checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)
    print "epochs %d" % epochs
    input_seq = Input(shape=(1, data_dim))
    # input_seq = Dropout(0.5)(input_seq)
    lstm1 = LSTM(lstm_dim,return_sequences=True)(input_seq)
    lstm2 = LSTM(lstm_dim,return_sequences=True)(lstm1)
    lstm3 = LSTM(lstm_dim)(lstm2)
    auxiliary_input = Input(shape=(aux_input_dim,), name='aux_input')
    concat_layer = keras.layers.concatenate([lstm3, auxiliary_input])

    d1 = Dense(lstm_dim, activation='relu')(concat_layer)
    d2 = Dense(lstm_dim, activation='relu')(d1)
    d2 = Dropout(0.5)(d2)
    out = Dense(1, activation='relu')(d2)

    model = Model(inputs=[input_seq,auxiliary_input], outputs=out)
    ada = Adadelta(lr=5.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['mse','accuracy'])
    lstm_layers = 3
    dnn_layers = 2
    name = str(lstm_layers)+"layer_lstm_"+str(dnn_layers)+"layer_dnn"
    print "start Training " + name
    save_path_of_pdf= save_path + name

    csv_path = save_path + name + ".csv"
    csv_file = open(csv_path,"w")
    csv_file.close()

    roc_path = save_path + name + "_roc"
    plot_roc = Plot_ROC(model_name=name, predictor=model, roc_path=roc_path,csv_path=csv_path,fig_path=save_path_of_pdf, x_test=[test_data, np.array(aux_input_data[2])], y_test=test_label)
    lrate = ReduceLROnPlateau(min_lr=0.00001)
    model.fit_generator(generate_arrays_of_train_aux(train_data, aux_input_data[0], train_label, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation_aux(validation_data, aux_input_data[1], validation_label, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[plot_roc,lrate])#,initial_epoch=28)

    # model_predictor(model_path,save_path,name,epoch_now,test_data,test_label)

    return 0


def train_and_test_model_with_dnn(data_dim,n_time_steps, save_path,train_data, validation_data, test_data, train_label, validation_label, test_label, class_weight = {0:1,1:1}):
    epochs = itt_range
    lstm_dim = 5
    batch_size = 8
    steps_per_epoch = 1#int(math.ceil(float(len(train_label))/float(batch_size)))
    validation_steps = int(math.ceil(float(len(validation_label))/float(batch_size)))
    model_path = save_path+"ckpt_dnn.h5"
    # checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)
    print "epochs %d" % epochs
    #data_dim * n_time_steps + 2
    input_seq = Input(shape=(2,))
    d1 = Dense(lstm_dim,activation='relu')(input_seq)
    d1 = Dropout(0.5)(d1)
    d2 = Dense(lstm_dim,activation='relu')(d1)
    d2 = Dropout(0.5)(d2)
    d3 = Dense(lstm_dim,activation='relu')(d2)
    d3 = Dropout(0.5)(d3)
    d4 = Dense(lstm_dim,activation='relu')(d3)
    d4 = Dropout(0.5)(d4)
    out = Dense(1,activation='relu')(d4)

    model = Model(inputs=input_seq, outputs=out)
    ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['mse','accuracy'])
    lstm_layers = 0
    dnn_layers = 4
    name = str(lstm_layers)+"layer_lstm_"+str(dnn_layers)+"layer_dnn"
    print "start Training" + name
    save_path_of_pdf= save_path + name

    csv_path = save_path + name + ".csv"
    csv_file = open(csv_path,"w")
    csv_file.close()

    roc_path = save_path + name + "_roc"
    plot_roc = Plot_ROC(model_name=name, predictor=model, roc_path=roc_path,csv_path=csv_path,fig_path=save_path_of_pdf, x_test=test_data, y_test=test_label)
    lrate = ReduceLROnPlateau(min_lr=0.00001)
    model.fit_generator(generate_arrays_of_train(train_data, train_label, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(validation_data, validation_label, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=[plot_roc,lrate])#,initial_epoch=28)

    return 0

def train_and_test_model_with_3layer_sdae(data_dim,n_time_steps,save_path,train_data, validation_data, test_data, train_label, validation_label, test_label,class_weight = {0:1,1:1}):
    epochs = itt_range
    epoch_now = 1
    lstm_dim = n_time_steps
    batch_size = 128
    steps_per_epoch = int(math.ceil(float(len(train_label))/float(batch_size)))
    validation_steps = int(math.ceil(float(len(validation_label))/float(batch_size)))
    model_path = save_path+"ckpt_lstm_"+str(epoch_now)+".h5"
    # checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)
    encoding_dim = 1
    data_origin_dim = n_time_steps*data_dim + 3
    print "epochs %d" % epochs
    input_seq = Input(shape=(2,))
    encoded = Dense(encoding_dim, activation='relu')(input_seq)
    encoded1 = Dense(encoding_dim, activation='relu')(encoded)
    encoded2 = Dense(encoding_dim, activation='relu')(encoded1)
    decoded = Dense(data_origin_dim, activation='relu')(encoded2)

    autoencoder = Model(input=input_seq, output=decoded)
    logistic_regression = Dense(1,activation='relu')(decoded)
    encoder = Model(input=input_seq, output=logistic_regression)
    autoencoder.compile(optimizer='adam', loss='mse')

    n_layers = 3
    name = str(n_layers)+"layers_sdae"
    print "start Training" + name
    save_path_of_pdf= save_path + name

    csv_path = save_path + name + ".csv"
    csv_file = open(csv_path,"w")
    csv_file.close()

    roc_path = save_path + name + "_roc"
    plot_roc = Plot_ROC(model_name=name, predictor=encoder, roc_path=roc_path,csv_path=csv_path,fig_path=save_path_of_pdf, x_test=test_data, y_test=test_label)
    lrate = ReduceLROnPlateau(min_lr=0.00001)

    autoencoder.fit_generator(generate_arrays_of_train(train_data, train_data, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(validation_data, validation_data, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[plot_roc,lrate])#,initial_epoch=28)

    # model_predictor(model_path,save_path,name,epoch_now,test_data,test_label)

    return 0


#SdAE
# def train_and_test_model_with_3layer_sdae(data_dim,n_time_steps,save_path,train_data, validation_data, test_data, train_label, validation_label, test_label,class_weight = {0:1,1:1}):
#     epochs = itt_range
#     batch_size = 16
#     steps_per_epoch = 100
#     validation_steps = int(math.ceil(float(len(validation_label))/float(batch_size)))
#     encoding_dim = 40
#
#     print "epochs %d" % epochs
#     input_seq = Input(shape=(n_time_steps*data_dim,))
#     encoded = Dense(encoding_dim, activation='relu')(input_seq)
#     encoded1 = Dense(encoding_dim, activation='relu')(encoded)
#     encoded2 = Dense(encoding_dim, activation='relu')(encoded1)
#     decoded = Dense(n_time_steps*data_dim, activation='sigmoid')(encoded2)
#     autoencoder = Model(input=input_seq, output=decoded)
#     logistic_regression = Dense(1,activation='sigmoid')(decoded)
#     encoder = Model(input=input_seq, output=logistic_regression)
#     autoencoder.compile(optimizer='adam', loss='mse')
#
#     n_layers = 3
#     name = str(n_layers)+"layers_sdae"
#     print "start Training" + name
#     save_path_of_pdf= save_path + name
#
#     csv_path = save_path + name + ".csv"
#     csv_file = open(csv_path,"w")
#     csv_file.close()
#
#     roc_path = save_path + name + "_roc"
#
#     plot_roc = Plot_ROC(model_name=name, predictor=encoder, roc_path=roc_path,csv_path=csv_path,fig_path=save_path_of_pdf, x_test=test_data, y_test=test_label)
#
#     autoencoder.fit_generator(generate_arrays_of_train(train_data, train_data, batch_size),
#     steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(validation_data, validation_data, batch_size),
#     validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=[plot_roc])
#     return 0

class Plot_ROC(keras.callbacks.Callback):
    def __init__(self, model_name, predictor, roc_path,csv_path, fig_path, x_test,y_test):
        super(Plot_ROC, self).__init__()

        self.model_name = model_name
        self.predictor = predictor
        self.csv_path = csv_path
        self.roc_path = roc_path
        self.fig_path = fig_path
        self.x_test = x_test
        self.y_test = y_test
        self.len_y = len(y_test)

    def on_epoch_end(self, epoch, logs={}):
        color = 'blue'
        lw = 2
        plt.clf()
        plt.cla()
        batch_size_of_test = 128
        test_steps = int(math.ceil(float(self.len_y)/float(batch_size_of_test)))
        print "I am in %d" % epoch
        # probas_ = self.model.predict_generator(generate_arrays_of_test(self.x_test,self.y_test,batch_size_of_test),test_steps)
        if self.predictor == None:
            probas_ = self.model.predict(self.x_test, batch_size=128)
        else:
            probas_ = self.predictor.predict(self.x_test, batch_size=128)
        pred = probas_[:, 0]
        indexs =self.y_test > 0
        positives = self.y_test[indexs]
        pred_pos = pred[indexs]

        print "I am out %d" % epoch

        mae = mean_absolute_error(positives, pred_pos)
        mre = mean_relative_error(positives, pred_pos)
        mse = mean_squared_error(positives, pred_pos)
        rmse = math.sqrt(mse)
        print "epoch: %d, %.4f, %.4f, %.4f" % (epoch, mae, mre, rmse)

        # fpr, tpr, thresholds = roc_curve(self.y_test, pred)
        # #
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=lw, color=color, label='ROC (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.title('ROC of ' + self.model_name + 'of epoch ' + str(epoch))
        # plt.legend(loc="lower right")
        # plt.savefig(self.fig_path+"_"+str(epoch)+".pdf")
        # #
        # pred_class = np.round(np.array(pred)).astype(int)
        # (precision,recall,fbeta_score,support)=precision_recall_fscore_support(self.y_test, pred_class,pos_label=1, average='binary')
        #
        # #
        # with open(self.csv_path, "a") as csv_file:
        #     ltw = ",".join(["epoch_"+str(epoch),str(precision), str(recall), str(fbeta_score), str(roc_auc)])
        #     print ltw
        #     csv_file.write(ltw + "\n")
        #
        # roc_path = self.roc_path + "_"+str(epoch)+".csv"
        # len_fpr = len(fpr)
        # out_str = ""
        # for itr in range(len_fpr):
        #     out_str += str(fpr[itr]) + "," + str(tpr[itr]) + "\n"
        # with open(roc_path,"w") as out_file:
        #     out_file.write(out_str)

def train_and_plot_roc(classifier, classifier_name, save_path, train_data, train_label, test_data, test_label):
    color = 'blue'
    lw = 2
    plt.clf()
    plt.cla()
    print "start fit %s" % classifier_name
    classifier.fit(train_data, np.array(train_label))
    print "finish fit %s" % classifier_name
    if classifier_name == "svm" or classifier_name == "svr" or classifier_name=="decision_tree_regressor":
        print "now predicting"
        probas_ = classifier.predict(test_data)
        pred = probas_
        # pred_class = pred
    else:
        probas_ = classifier.predict_proba(test_data)
        pred = np.array(probas_[:, 1])
        # pred_class = classifier.predict(test_data)
        # pred_class = np.round(np.array(pred)).astype(int)
        # super_threshold_indices = pred > 0.7
        # sub_threshold_indices = pred <= 0.7
        # pred_class = pred
        # pred_class[super_threshold_indices] = 1
        # pred_class[sub_threshold_indices] = 0
        # pred_class = pred_class.astype(int)
    print "finish predict %s" % classifier_name

    indexs = test_label > 0
    positives = test_label[indexs]
    pred_pos = pred[indexs]

    mae = mean_absolute_error(positives, pred_pos)
    mre = mean_relative_error(positives, pred_pos)
    mse = mean_squared_error(positives, pred_pos)
    rmse = math.sqrt(mse)
    print "%.4f, %.4f, %.4f" %(mae, mre, rmse)
    return (mae, mre, rmse)
    # fpr, tpr, thresholds = roc_curve(test_label, pred)
    # roc_auc = auc(fpr, tpr)
    # title="ROC of "+ classifier_name + ": auc " + str(round(roc_auc, 2))
    # pdf_path = save_path + classifier_name + ".pdf"
    # figure, axes = plot_roc(tpr, fpr, thresholds, label_every=1000, title= title)
    # plt.plot(fpr, tpr, lw=lw, color=color,
    #     label='ROC (area = %0.2f)' % (roc_auc))

    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic of ' + classifier_name)
    # plt.legend(loc="lower right")
    # plt.savefig(pdf_path)

    #write precision, recall, etc
    # (precision,recall,fbeta_score,support)=precision_recall_fscore_support(test_label, pred_class, pos_label=1, average='binary')
    # precision_path = save_path + classifier_name + ".csv"
    # with open(precision_path, "w") as csv_file:
    #     ltw = ",".join([str(precision), str(recall), str(fbeta_score), str(roc_auc)])
    #     print classifier_name + " " + ltw
    #     csv_file.write(ltw + "\n")

    # len_fpr = len(fpr)
    # out_str = ""
    # for itr in range(len_fpr):
    #     out_str += str(fpr[itr]) + "," + str(tpr[itr]) + "\n"
    #
    # tpr_fpr_path = save_path + classifier_name + "_roc.csv"
    # with open(tpr_fpr_path,"w") as outfile:
    #     outfile.write(out_str)
    # print "write succ of %s" % classifier_name

#lasso_regression
def train_and_test_model_with_lasso_regression(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    random_state = np.random.RandomState(0)
    lasso = LogisticRegression(C=0.00001, penalty='l1',solver='liblinear', tol=1e-4,class_weight='balanced', n_jobs=-1,random_state=random_state)
    train_and_plot_roc(lasso,'lasso_regression C=0.00001',save_path,train_data, train_label, test_data, test_label)

    del lasso

    lasso = LogisticRegression(C=0.1, penalty='l1',solver='liblinear', tol=1e-4,class_weight='balanced', n_jobs=-1,random_state=random_state)
    train_and_plot_roc(lasso,'lasso_regression C=0.1',save_path,train_data, train_label, test_data, test_label)

    # lasso = LogisticRegression(C=1.0, penalty='l1',solver='liblinear', tol=1e-4,class_weight='balanced', n_jobs=-1,random_state=random_state)
    # train_and_plot_roc(lasso,'lasso_regression C=1',save_path,train_data, train_label, test_data, test_label)

    # lasso = LogisticRegression(C=10.0, penalty='l1',solver='liblinear', tol=1e-4,class_weight='balanced', n_jobs=-1,random_state=random_state)
    # train_and_plot_roc(lasso,'lasso_regression C=10',save_path,train_data, train_label, test_data, test_label)
#ridge_regression
def train_and_test_model_with_ridge_regression(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    random_state = np.random.RandomState(0)
    ridge = LogisticRegression(C=10.0, penalty='l2',solver='sag', tol=1e-4,class_weight='balanced', n_jobs=-1,random_state=random_state)

    train_and_plot_roc(ridge,'ridge_regression',save_path,train_data, train_label, test_data, test_label)
    #
    # grid_search =GridSearchCV(LogisticRegression(penalty='l2',solver='sag', tol=1e-4,class_weight='balanced', n_jobs=-1), param_grid=tuned_parameters, cv=5, scoring='f1')#['accuracy','precision','recall','f1'])
    # print("Performing grid search...")
    # print("pipeline:LogisticRegression")
    # print("parameters:")
    # print(tuned_parameters)
    # t0 = time()
    #
    # # 这里只需调用一次fit函数就可以了
    # grid_search.fit(all_data_list, all_label_list)
    # print("done in %.2f s" % (time() - t0))
    #
    # # 输出best score
    # print("Best score: %.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    #
    # # 输出最佳的分类器到底使用了怎样的参数
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(tuned_parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

#RandomForest
def train_and_test_model_with_random_forest(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "RandomForestClassifier"
    rfc = RandomForestClassifier(n_jobs=-1,criterion='entropy',n_estimators=50,class_weight='balanced')
    train_and_plot_roc(rfc,'random_forest',save_path,train_data, train_label, test_data, test_label)
#LinearSVC
def train_and_test_model_with_svm(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "LinearSVC"
    svc = LinearSVC(C=0.1,class_weight='balanced')
    train_and_plot_roc(svc,'svm',save_path,train_data, train_label, test_data, test_label)
def train_and_test_model_with_svr(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "LinearSVC"
    svc = SVR(C=0.00001)
    (mae, mre, rmse) = train_and_plot_roc(svc,'svr',save_path,train_data, train_label, test_data, test_label)
    return (mae, mre, rmse)
#DecisionTree
def train_and_test_model_with_decision_tree(data_dim,n_time_steps, train_data, train_label, test_data, test_label,save_path):
    # print "DecisionTreeClassifier"
    dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
    train_and_plot_roc(dtc,'decision_tree',save_path,train_data, train_label, test_data, test_label)

def train_and_test_model_with_decision_tree_regressor(data_dim,n_time_steps, train_data, train_label, test_data, test_label,save_path):
    # print "DecisionTreeClassifier"
    dtr = DecisionTreeRegressor(max_depth=5)
    train_and_plot_roc(dtr,'decision_tree_regressor',save_path,train_data, train_label, test_data, test_label)

#LDA
def train_and_test_model_with_lda(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "LinearDiscriminantAnalysis"
    lda = LinearDiscriminantAnalysis(solver='svd',tol=1e-4)
    train_and_plot_roc(lda,'lda',save_path,train_data, train_label, test_data, test_label)
#QDA
def train_and_test_model_with_qda(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    qda = QuadraticDiscriminantAnalysis(tol=1e-4)
    # print "QuadraticDiscriminantAnalysis"
    train_and_plot_roc(qda,'qda',save_path,train_data, train_label, test_data, test_label)
#AdaBoost
def train_and_test_model_with_ada_boost(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "AdaBoostClassifier"
    ada_boost = AdaBoostClassifier(learning_rate=1.0, n_estimators=50)
    train_and_plot_roc(ada_boost,'ada_boost',save_path,train_data, train_label, test_data, test_label)
#GradientBoostingClassifier
def train_and_test_model_with_gradient_boosting(data_dim,n_time_steps,train_data, train_label, test_data, test_label,save_path):
    # print "GradientBoostingClassifier"
    gbc = GradientBoostingClassifier(learning_rate=1.0, n_estimators=50)
    train_and_plot_roc(gbc,'gradient_boosting',save_path,train_data, train_label, test_data, test_label)

