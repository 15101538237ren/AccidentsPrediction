# -*- coding: utf-8 -*-
from time import time
import keras,math,sys
import numpy as np
from scipy import interp
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
from util import generate_arrays_of_validation,generate_arrays_of_train
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,NuSVC
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error,precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
reload(sys)
sys.setdefaultencoding('utf8')

mean_mae_of_lstm = [[] for itt in range(16)]
mean_mse_of_lstm = [[] for itt in range(16)]
mean_rmse_of_lstm = [[] for itt in range(16)]
mean_precision_of_lstm = [[] for itt in range(16)]
mean_recall_of_lstm = [[] for itt in range(16)]
mean_fscore_of_lstm = [[] for itt in range(16)]
mean_auc_of_lstm = [[] for itt in range(16)]

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#Dense Network
def train_and_test_model_with_dense_network(data_dim,n_time_steps, all_data_list, all_label_list,save_path ,split_ratio=0.8,class_weight={0:1,1:1}):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_1_layer_dense_network.h5", verbose=1)

    batch_size = 256
    steps_per_epoch = 500
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs

    input_seq = Input(shape=(n_time_steps*data_dim,))
    x = Dense(n_time_steps * 2,activation='relu')(input_seq)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training 1 Layer Dense Network model"

    save_path_of_pdf= save_path + "roc_of_1_layer_dense_network"
    plot_roc = Plot_ROC(model_name='1 layer dense network',fig_path=save_path_of_pdf, x_test=all_val_data_list, y_test=all_val_label_list)
    callbacks = [lrate, early_stoping, checkpointer,plot_roc]

    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)
    # return 0

#2 Layer Dense Network
def train_and_test_model_with_2_layer_dense_network(data_dim,n_time_steps, all_data_list, all_label_list,save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_2_layer_dense_network.h5", verbose=1)

    batch_size = 256
    steps_per_epoch = 500
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs

    input_seq = Input(shape=(n_time_steps*data_dim,))
    x = Dense(n_time_steps * 2,activation='relu')(input_seq)
    x = Dropout(0.4)(x)
    x = Dense(n_time_steps,activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training 2 Layer Dense Network model"
    save_path_of_pdf= save_path + "roc_of_2_layer_dense_network"

    plot_roc = Plot_ROC(model_name='2 layer dense network',fig_path=save_path_of_pdf, x_test=all_val_data_list, y_test= all_val_label_list)
    callbacks = [lrate, early_stoping, checkpointer, plot_roc]

    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)
    return 0

def train_and_test_model_with_sdae(data_dim,n_time_steps,all_data_list, all_label_list,save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    classifier_name = "SdAE"
    cv = StratifiedKFold(n_splits=5)
    it = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])

    lw = 2
    i = 0
    plt.clf()
    plt.cla()
    mean_mae = []
    mean_mse = []
    mean_rmse = []
    mean_precision = []
    mean_recall = []
    mean_fscore = []
    for (train, test), color in zip(cv.split(all_data_list, all_label_list), colors):
        all_val_label_list = all_label_list[test]
        all_train_label_list = all_label_list[train]

        all_val_data_list = all_data_list[test]
        all_train_data_list = all_data_list[train]

        it += 1

        batch_size = 16
        steps_per_epoch = 1000
        epochs = 2
        validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))

        encoding_dim=40
        input_seq = Input(shape=(n_time_steps*data_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_seq)
        decoded = Dense(n_time_steps*data_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_seq, output=decoded)
        logistic_regression = Dense(1,activation='sigmoid')(encoded)
        
        encoder = Model(input=input_seq, output=logistic_regression)
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
        steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
        validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=[])
        
        pred_class = encoder.predict(all_val_data_list)
        fpr, tpr, thresholds = roc_curve(all_val_label_list, pred_class)
        (precision,recall,fbeta_score,support)=precision_recall_fscore_support(all_val_label_list, pred_class, pos_label =1, average='binary')
        
        mean_precision.append(precision)
        mean_recall.append(recall)
        mean_fscore.append(fbeta_score)

        t_mae = mean_absolute_error(all_val_label_list, pred_class)
        mean_mae.append(t_mae)
        t_mse = mean_squared_error(all_val_label_list, pred_class)
        mean_mse.append(t_mse)
        t_rmse = np.sqrt(mean_squared_error(all_val_label_list, pred_class))
        mean_rmse.append(t_rmse)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

    mean_tpr /= cv.get_n_splits(all_data_list, all_label_list)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    print "\t".join("SdAE:",[str(np.mean(np.array(mean_mae))), str(np.mean(np.array(mean_mse))), str(np.mean(np.array(mean_rmse))),str(np.mean(np.array(mean_precision))),str(np.mean(np.array(mean_recall))),str(np.mean(np.array(mean_fscore))),str(mean_auc)])

    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of ' + classifier_name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(save_path+"sdae.pdf")

#LSTM
def train_and_test_model_with_lstm(data_dim,n_time_steps,all_data_list, all_label_list,save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    cv = StratifiedKFold(n_splits=5)
    it = 0
    epochs = 15 #int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2000

    
    for (train, test) in cv.split(all_data_list, all_label_list):
        all_val_label_list = all_label_list[test]
        all_train_label_list = all_label_list[train]

        all_val_data_list = all_data_list[test]
        all_train_data_list = all_data_list[train]

        it += 1
        lrate = ReduceLROnPlateau(min_lr=0.00001)
        early_stoping = EarlyStopping(monitor='val_loss',patience=10)
        checkpointer = ModelCheckpoint(filepath="ckpt_1_layer_lstm.h5", verbose=1)

        lstm_dim = n_time_steps
        batch_size = 16
        steps_per_epoch = 100
        validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))

        print "epochs %d" % epochs

        input_seq = Input(shape=(n_time_steps,data_dim))#shape=(n_time_steps,data_dim))
        lstm1 = LSTM(lstm_dim)(input_seq)
        out = Dense(1,activation='sigmoid')(lstm1)

        model = Model(inputs=input_seq, outputs=out)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

        print "start Training LSTM model %d" % it
        save_path_of_pdf= save_path + "roc_of_1_layer_lstm"

        plot_roc = Plot_ROC_CV_OF_LSTM(model_name='1 layer lstm model',fig_path=save_path_of_pdf,cv_id=it, x_test=all_val_data_list, y_test=all_val_label_list)
        callbacks = [lrate, checkpointer,plot_roc,early_stoping]

        model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
        steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
        validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)

    for epoch in range(1, epochs):
        print "\t".join("lstm epoch "+str(epoch)+":",[str(np.mean(np.array(mean_mae_of_lstm[epoch]))), str(np.mean(np.array(mean_mse_of_lstm[epochs]))), str(np.mean(np.array(mean_rmse_of_lstm[epoch]))),str(np.mean(np.array(mean_precision_of_lstm[epoch]))),str(np.mean(np.array(mean_recall_of_lstm[epoch]))),str(np.mean(np.array(mean_fscore_of_lstm[epoch]))),str(mean_auc_of_lstm[epoch])])
    return 0
#LSTM
def train_and_test_model_with_2_layer_lstm(data_dim,n_time_steps,all_data_list, all_label_list,save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))


    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_2_layer_lstm.h5", verbose=1)

    lstm_dim = n_time_steps
    batch_size = 16
    steps_per_epoch = 100
    epochs = 15 #int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2000
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))

    print "epochs %d" % epochs

    input_seq = Input(shape=(n_time_steps,data_dim))#shape=(n_time_steps,data_dim))
    lstm1 = LSTM(output_dim= lstm_dim,return_sequences=True)(input_seq)
    lstm2 = LSTM(output_dim=lstm_dim,return_sequences=False)(lstm1)
    lstm2 = Dropout(0.5)(lstm2)
    out = Dense(1,activation='sigmoid')(lstm2)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training 2 layer LSTM model"
    save_path_of_pdf= save_path + "roc_of_2_layer_lstm"

    plot_roc = Plot_ROC(model_name='2 layer lstm model',fig_path=save_path_of_pdf, x_test=all_val_data_list, y_test=all_val_label_list)
    callbacks = [lrate, checkpointer,plot_roc,early_stoping]

    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)
    return 0
#GRU
def train_and_test_model_with_gru(data_dim,n_time_steps,all_data_list, all_label_list,save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    # tot_len = len(all_label_list)
    # val_n = int(tot_len * (1.0 - split_ratio))
    #
    # all_val_label_list = all_label_list[0:val_n]
    # all_train_label_list = all_label_list[val_n:-1]
    #
    # all_val_data_list = all_data_list[0:val_n]
    # all_train_data_list = all_data_list[val_n:-1]
    cv = StratifiedKFold(n_splits=5)
    it = 0
    for (train, test) in cv.split(all_data_list, all_label_list):
        all_val_label_list = all_label_list[test]
        all_train_label_list = all_label_list[train]

        all_val_data_list = all_data_list[test]
        all_train_data_list = all_data_list[train]

        it += 1

        lrate = ReduceLROnPlateau(min_lr=0.00001)
        early_stoping = EarlyStopping(monitor='val_loss',patience=10)
        checkpointer = ModelCheckpoint(filepath="ckpt_1_layer_gru.h5", verbose=1)

        gru_dim = n_time_steps
        batch_size = 16
        steps_per_epoch = 100
        epochs = 15#int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2000
        validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
        print "epochs %d" % epochs

        input_seq = Input(shape=(n_time_steps,data_dim))
        gru = GRU(gru_dim)(input_seq)
        gru = Dropout(0.5)(gru)
        out = Dense(1,activation='sigmoid')(gru)

        model = Model(inputs=input_seq, outputs=out)
        optimizer = RMSprop(clipvalue=0.5,clipnorm=1.)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mse','accuracy'])

        print "start Training GRU model of %d" % it
        save_path_of_pdf= save_path + "roc_of_1_layer_gru"

        plot_roc = Plot_ROC_CV(model_name='gru model',fig_path=save_path_of_pdf,cv_id=it, x_test=all_val_data_list,y_test=all_val_label_list)
        callbacks = [lrate, checkpointer,plot_roc]

        model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
        steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
        validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)
    return 0

#keras_logistic_regression
def train_and_test_model_with_keras_logistic_regression(data_dim,n_time_steps,all_data_list, all_label_list, save_path,split_ratio=0.8,class_weight={0:1,1:1}):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    # rus = RandomUnderSampler(random_state=42)
    # X_res, y_res = rus.fit_sample(all_data_list, all_label_list)
    # print('Resampled dataset shape {}'.format(Counter(y_res)))
    # int_ratio = int(0.4 * len(y_res))
    # x_test,y_test = X_res[0:int_ratio],y_res[0:int_ratio]
    #
    #
    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_logistic_regression.h5", verbose=1)

    batch_size = 256
    steps_per_epoch = 1000
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs

    input_seq = Input(shape=(n_time_steps*data_dim,))
    out = Dense(1, activation='sigmoid')(input_seq)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training Keras Logistic Regression model"
    save_path_of_pdf= save_path + "roc_of_keras_logistic_regression"
    plot_roc = Plot_ROC(model_name='keras logistic regression',fig_path=save_path_of_pdf, x_test=all_val_data_list, y_test=all_val_label_list)
    callbacks = [lrate, early_stoping, checkpointer, plot_roc]


    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1,class_weight=class_weight, callbacks=callbacks)#, initial_epoch=28)
    return 0
class Plot_ROC_CV(keras.callbacks.Callback):
    def __init__(self, model_name, fig_path,cv_id, x_test,y_test):
        super(Plot_ROC_CV, self).__init__()
        self.model_name = model_name
        self.fig_path = fig_path
        self.x_test = x_test
        self.y_test = y_test
        self.cv_id = cv_id

    def on_epoch_end(self, epoch, logs={}):
        color = 'blue'
        lw = 2
        plt.clf()
        plt.cla()

        probas_ = self.model.predict(self.x_test)
        pred = probas_[:, 0]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(self.y_test, pred)
        print "MAE of " + self.model_name+ ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : " + str(mean_absolute_error(self.y_test, pred))
        # print "MAPE of " + self.model_name + "of epoch " + str(epoch) +" : " + str(mean_absolute_percentage_error(self.y_test, pred))
        print "MSE of " + self.model_name + ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : "  + str(mean_squared_error(self.y_test, pred))
        print "RMSE of" + self.model_name + ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : "  + str(np.sqrt(mean_squared_error(self.y_test, pred)))

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of ' + self.model_name + 'of epoch ' + str(epoch) + ' of ' + str(self.cv_id))
        plt.legend(loc="lower right")
        plt.savefig(self.fig_path+"_"+str(self.cv_id)+"_"+str(epoch)+".pdf")


class Plot_ROC_CV_OF_LSTM(keras.callbacks.Callback):
    def __init__(self, model_name, fig_path,cv_id, x_test,y_test):
        super(Plot_ROC_CV_OF_LSTM, self).__init__()
        self.model_name = model_name
        self.fig_path = fig_path
        self.x_test = x_test
        self.y_test = y_test
        self.cv_id = cv_id

    def on_epoch_end(self, epoch, logs={}):
        color = 'blue'
        lw = 2
        plt.clf()
        plt.cla()

        probas_ = self.model.predict(self.x_test)
        pred = probas_[:, 0]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(self.y_test, pred)
        print "MAE of " + self.model_name+ ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : " + str(mean_absolute_error(self.y_test, pred))
        print "MSE of " + self.model_name + ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : "  + str(mean_squared_error(self.y_test, pred))
        print "RMSE of" + self.model_name + ' of ' + str(self.cv_id) + " of epoch " + str(epoch) +" : "  + str(np.sqrt(mean_squared_error(self.y_test, pred)))
        mean_mae_of_lstm[epoch].append(mean_absolute_error(self.y_test, pred))
        mean_mse_of_lstm[epoch].append(mean_squared_error(self.y_test, pred))
        mean_rmse_of_lstm[epoch].append(np.sqrt(mean_squared_error(self.y_test, pred)))

        (precision,recall,fbeta_score,support)=precision_recall_fscore_support(self.y_test, pred, pos_label =1, average='binary')
        mean_precision_of_lstm[epoch].append(precision)
        mean_recall_of_lstm[epoch].append(recall)
        mean_fscore_of_lstm[epoch].append(fbeta_score)
        roc_auc = auc(fpr, tpr)
        mean_auc_of_lstm[epoch].append(roc_auc)

        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of ' + self.model_name + 'of epoch ' + str(epoch) + ' of ' + str(self.cv_id))
        plt.legend(loc="lower right")
        plt.savefig(self.fig_path+"_"+str(self.cv_id)+"_"+str(epoch)+".pdf")
class Plot_ROC(keras.callbacks.Callback):
    def __init__(self, model_name, fig_path, x_test,y_test):
        super(Plot_ROC, self).__init__()
        self.model_name = model_name
        self.fig_path = fig_path
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        color = 'red'
        lw = 2
        plt.clf()
        plt.cla()

        probas_ = self.model.predict(self.x_test)
        pred = probas_[:, 0]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(self.y_test, pred)
        print "MAE of " + self.model_name + "of epoch " + str(epoch) +" : " + str(mean_absolute_error(self.y_test, pred))
        # print "MAPE of " + self.model_name + "of epoch " + str(epoch) +" : " + str(mean_absolute_percentage_error(self.y_test, pred))
        print "MSE of " + self.model_name + "of epoch " + str(epoch) +" : "  + str(mean_squared_error(self.y_test, pred))
        print "RMSE of" + self.model_name + "of epoch " + str(epoch) +" : "  + str(np.sqrt(mean_squared_error(self.y_test, pred)))

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC (area = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of ' + self.model_name + 'of epoch ' + str(epoch))
        plt.legend(loc="lower right")
        plt.savefig(self.fig_path+"_"+str(epoch)+".pdf")

def train_and_plot_roc(classifier, classifier_name, save_path, X, y):
    cv = StratifiedKFold(n_splits=5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])

    lw = 2
    i = 0
    plt.clf()
    plt.cla()
    mean_mae = []
    # mean_mape = []
    mean_mse = []
    mean_rmse = []
    mean_precision = []
    mean_recall = []
    mean_fscore = []

    for (train, test), color in zip(cv.split(X, y), colors):

        classifier.fit(X[train], y[train])
        # print "finish fit"
        if classifier_name == "svm":
            probas_ = classifier.predict(X[test])
            pred = probas_
            pred_class = pred
        else:
            probas_ = classifier.predict_proba(X[test])
            pred = probas_[:, 1]
            pred_class = classifier.predict(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], pred)

        (precision,recall,fbeta_score,support)=precision_recall_fscore_support(y[test], pred_class, pos_label =1, average='binary')

        mean_precision.append(precision)
        mean_recall.append(recall)
        mean_fscore.append(fbeta_score)

        t_mae = mean_absolute_error(y[test], pred)
        mean_mae.append(t_mae)
        # t_mape = mean_absolute_percentage_error(y[test], pred)
        # mean_mape.append(t_mape)
        t_mse = mean_squared_error(y[test], pred)
        mean_mse.append(t_mse)
        t_rmse = np.sqrt(mean_squared_error(y[test], pred))
        mean_rmse.append(t_rmse)


        # print "MAE of " + classifier_name + " of test " + str(i) +" : " + str(t_mae)
        # print "MAPE of " + classifier_name + " of test " + str(i) +" : "  + str(t_mape)
        # print "MSE of " + classifier_name + " of test " + str(i) +" : "  + str(t_mse)
        # print "RMSE of" + classifier_name + " of test " + str(i) +" : "  + str(t_rmse)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

    # print "MAE of " + classifier_name +" : " + str(np.mean(np.array(mean_mae)))
    # print "MSE of " + classifier_name +" : "  + str(np.mean(np.array(mean_mse)))
    # print "RMSE of " + classifier_name +" : "  + str(np.mean(np.array(mean_rmse)))


    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    print "\t".join([str(np.mean(np.array(mean_mae))), str(np.mean(np.array(mean_mse))), str(np.mean(np.array(mean_rmse))),str(np.mean(np.array(mean_precision))),str(np.mean(np.array(mean_recall))),str(np.mean(np.array(mean_fscore))),str(mean_auc)])

    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of ' + classifier_name)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(save_path)

#logistic_regression
def train_and_test_model_with_logistic_regression(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # tuned_parameters = {'C': [1,10,100,1000]}
    random_state = np.random.RandomState(0)
    log_reg = LogisticRegression(C=10.0, penalty='l2',solver='sag', tol=1e-4,class_weight=class_weight, n_jobs=-1,random_state=random_state)

    # print "start Training Traditional Logistic Regression model"
    save_path_of_pdf= save_path + "roc_of_logistic_regression.pdf"
    train_and_plot_roc(log_reg,'logistic_regression',save_path_of_pdf,X,y)

    #grid_search =GridSearchCV(LogisticRegression(penalty='l2',solver='sag', tol=1e-4,class_weight='balanced', n_jobs=-1), param_grid=tuned_parameters, cv=5, scoring='f1')#['accuracy','precision','recall','f1'])
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


    # acc_scores = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    # print "acc_scores:",
    # print acc_scores
    #
    # precision = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='precision')
    # print "precision:",
    # print precision
    #
    # recall = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='recall')
    # print "recall:",
    # print recall
    #
    # f1 = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='f1')
    # print "f1:",
    # print f1
    #
    # roc_auc = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    # print "roc_auc:",
    # print roc_auc
    #
    # print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#LDA
def train_and_test_model_with_lda(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # print "LinearDiscriminantAnalysis"
    lda = LinearDiscriminantAnalysis(solver='svd',tol=1e-4)
    save_path_of_pdf= save_path + "roc_of_lda.pdf"
    train_and_plot_roc(lda,'lda',save_path_of_pdf,X,y)
#QDA
def train_and_test_model_with_qda(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7):
    qda = QuadraticDiscriminantAnalysis(tol=1e-4)
    # print "QuadraticDiscriminantAnalysis"
    save_path_of_pdf= save_path + "roc_of_qda.pdf"
    train_and_plot_roc(qda,'qda',save_path_of_pdf,X,y)

#AdaBoost
def train_and_test_model_with_ada_boost(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7):
    # print "AdaBoostClassifier"
    ada_boost = AdaBoostClassifier(learning_rate=1.0, n_estimators=50)

    save_path_of_pdf= save_path + "roc_of_ada_boost.pdf"
    train_and_plot_roc(ada_boost,'ada_boost',save_path_of_pdf,X,y)
#Bagging
def train_and_test_model_with_bagging(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7):
    # print "BaggingClassifier"
    bc = BaggingClassifier(n_jobs=-1, n_estimators=50)
    save_path_of_pdf= save_path + "roc_of_bagging.pdf"
    train_and_plot_roc(bc,'BaggingClassifier',save_path_of_pdf,X,y)
#RandomForest
def train_and_test_model_with_random_forest(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # print "RandomForestClassifier"
    rfc = RandomForestClassifier(n_jobs=-1,criterion='entropy',n_estimators=50,class_weight=class_weight)
    save_path_of_pdf= save_path + "roc_of_random_forest.pdf"
    train_and_plot_roc(rfc,'random_forest',save_path_of_pdf,X,y)
#GradientBoostingClassifier
def train_and_test_model_with_gradient_boosting(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7):
    # print "GradientBoostingClassifier"
    gbc = GradientBoostingClassifier(learning_rate=1.0, n_estimators=50)
    save_path_of_pdf= save_path + "roc_of_gradient_boosting.pdf"
    train_and_plot_roc(gbc,'gradient_boosting',save_path_of_pdf,X,y)
#ExtraTreesClassifier
def train_and_test_model_with_extra_tree(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # print "ExtraTreesClassifier"
    etc = ExtraTreesClassifier(n_jobs=-1,criterion='entropy',n_estimators=50,class_weight=class_weight)
    save_path_of_pdf= save_path + "roc_of_extra_tree.pdf"
    train_and_plot_roc(etc,'extra_tree',save_path_of_pdf,X,y)
#LinearSVC
def train_and_test_model_with_svm(data_dim,n_time_steps,X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # print "LinearSVC"
    svc = LinearSVC(C=10.0,class_weight=class_weight)
    save_path_of_pdf= save_path + "roc_of_svm.pdf"
    train_and_plot_roc(svc,'svm',save_path_of_pdf,X,y)
#DecisionTree
def train_and_test_model_with_decision_tree(data_dim,n_time_steps, X, y,save_path,split_ratio=0.7,class_weight={0:1,1:1}):
    # print "DecisionTreeClassifier"
    dtc = DecisionTreeClassifier(max_depth=10,class_weight=class_weight)
    save_path_of_pdf= save_path + "roc_of_decision_tree.pdf"
    train_and_plot_roc(dtc,'decision_tree',save_path_of_pdf,X,y)