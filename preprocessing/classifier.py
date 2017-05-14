# -*- coding: utf-8 -*-
from time import time
import keras,math,sys
from keras.models import Model,load_model,Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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
reload(sys)
sys.setdefaultencoding('utf8')

#Dense Network
def train_and_test_model_with_dense_network(n_time_steps, all_data_list, all_label_list,split_ratio=0.8):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]
    print "all_train_data_list[0].shape"
    print all_train_data_list[0].shape

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_1_layer_dense_network.h5", verbose=1)

    data_dim = 4
    batch_size = 256
    steps_per_epoch = 10000
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 3
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs

    input_seq = Input(shape=(n_time_steps*data_dim,))
    x = Dense(n_time_steps * 2,activation='relu')(input_seq)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training 1 Layer Dense Network model"
    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate,early_stoping])#, initial_epoch=28)
    return 0

#2 Layer Dense Network
def train_and_test_model_with_2_layer_dense_network(n_time_steps, all_data_list, all_label_list,split_ratio=0.8):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_2_layer_dense_network.h5", verbose=1)

    data_dim = 4
    batch_size = 256
    steps_per_epoch = 10000
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 3
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
    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate,early_stoping])#, initial_epoch=28)
    return 0

#LSTM
def train_and_test_model_with_lstm(n_time_steps,all_data_list, all_label_list,split_ratio=0.8):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_1_layer_lstm.h5", verbose=1)

    lstm_dim = n_time_steps
    batch_size = 256
    steps_per_epoch = 10000
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs
    data_dim = 4
    input_seq = Input(shape=(n_time_steps,data_dim))
    lstm1 = LSTM(lstm_dim)(input_seq)
    out = Dense(1,activation='sigmoid')(lstm1)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training LSTM model"
    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate,early_stoping])#, initial_epoch=28)
    return 0
#keras_logistic_regression
def train_and_test_model_with_keras_logistic_regression(n_time_steps,all_data_list, all_label_list,split_ratio=0.8):
    tot_len = len(all_label_list)
    val_n = int(tot_len * (1.0 - split_ratio))

    all_val_label_list = all_label_list[0:val_n]
    all_train_label_list = all_label_list[val_n:-1]

    all_val_data_list = all_data_list[0:val_n]
    all_train_data_list = all_data_list[val_n:-1]

    print "all_train_data_list[0].shape"
    print all_train_data_list[0].shape
    lrate = ReduceLROnPlateau(min_lr=0.00001)
    early_stoping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath="ckpt_logistic_regression.h5", verbose=1)

    batch_size = 256
    steps_per_epoch = 10000
    epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size))) * 2
    validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    print "epochs %d" % epochs

    data_dim = 4
    input_seq = Input(shape=(n_time_steps*data_dim,))
    out = Dense(1, activation='sigmoid')(input_seq)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['mse','accuracy'])

    print "start Training Keras Logistic Regression model"
    model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size),
    validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate,early_stoping])#, initial_epoch=28)
    return 0

#logistic_regression
def train_and_test_model_with_logistic_regression(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'C': [1,10,100,1000]}

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

    log_reg = LogisticRegression(C=10.0, penalty='l2',solver='sag', tol=1e-4,class_weight='balanced', n_jobs=-1)
    acc_scores = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(log_reg,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#LDA
def train_and_test_model_with_lda(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'shrinkage': [0.0, 0.3, 0.6, 1.0], 'solver':['lsqr','eigen']}
    # grid_search =GridSearchCV(LinearDiscriminantAnalysis(tol=1e-4), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: LinearDiscriminantAnalysis")
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

    print "LinearDiscriminantAnalysis"
    lda = LinearDiscriminantAnalysis(solver='svd',tol=1e-4)
    acc_scores = cross_val_score(lda,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(lda,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(lda,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(lda,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(lda,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#QDA
def train_and_test_model_with_qda(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    qda = QuadraticDiscriminantAnalysis(tol=1e-4)
    print "QuadraticDiscriminantAnalysis"
    acc_scores = cross_val_score(qda,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(qda,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(qda,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(qda,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(qda,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())


#AdaBoost
def train_and_test_model_with_ada_boost(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'n_estimators':[50, 100]}
    # grid_search =GridSearchCV(AdaBoostClassifier(learning_rate=1.0), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: AdaBoostClassifier")
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

    print "AdaBoostClassifier"
    ada_boost = AdaBoostClassifier(learning_rate=1.0, n_estimators=100)
    acc_scores = cross_val_score(ada_boost,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(ada_boost,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(ada_boost,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(ada_boost,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(ada_boost,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#Bagging
def train_and_test_model_with_bagging(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'n_estimators':[10, 20, 50]}
    #
    # grid_search =GridSearchCV(BaggingClassifier(n_jobs=-1), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: BaggingClassifier")
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

    print "BaggingClassifier"
    bc = BaggingClassifier(n_jobs=-1, n_estimators=100)
    acc_scores = cross_val_score(bc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(bc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(bc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(bc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(bc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#RandomForest
def train_and_test_model_with_random_forest(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'n_estimators':[50,100]}
    #
    # grid_search =GridSearchCV(RandomForestClassifier(n_jobs=-1,criterion='entropy'), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: RandomForestClassifier")
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

    print "RandomForestClassifier"
    rfc = RandomForestClassifier(n_jobs=-1,criterion='entropy',n_estimators=100)

    acc_scores = cross_val_score(rfc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(rfc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(rfc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(rfc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(rfc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#GradientBoostingClassifier
def train_and_test_model_with_gradient_boosting(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'learning_rate':[0.1, 0.5, 1.0],'n_estimators':[10, 25, 50]}
    #
    # grid_search =GridSearchCV(GradientBoostingClassifier(), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: GradientBoostingClassifier")
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

    print "GradientBoostingClassifier"
    gbc = GradientBoostingClassifier(learning_rate=1.0, n_estimators=100)
    acc_scores = cross_val_score(gbc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(gbc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(gbc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(gbc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(gbc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#ExtraTreesClassifier
def train_and_test_model_with_extra_tree(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'criterion':['gini','entropy'],'n_estimators':[10, 25, 50]}
    #
    # grid_search =GridSearchCV(ExtraTreesClassifier(n_jobs=-1), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: ExtraTreesClassifier")
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

    print "ExtraTreesClassifier"
    etc = ExtraTreesClassifier(n_jobs=-1,criterion='entropy',n_estimators=100)
    acc_scores = cross_val_score(etc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(etc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(etc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(etc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(etc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#LinearSVC
def train_and_test_model_with_svm(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'C':[5.0, 10.0]}
    #
    # grid_search =GridSearchCV(LinearSVC(), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: LinearSVC")
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

    print "LinearSVC"
    svc = LinearSVC(C=10.0)
    acc_scores = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#rbf_nu_svm
def train_and_test_model_with_rbf_nu_svm(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'C':[5.0, 10.0]}
    #
    # grid_search =GridSearchCV(LinearSVC(), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: LinearSVC")
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

    print "NuSVC"
    svc = NuSVC()
    acc_scores = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(svc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())

#DecisionTree
def train_and_test_model_with_decision_tree(n_time_steps,all_data_list, all_label_list,split_ratio=0.7):
    # tuned_parameters = {'max_depth':[4, 6, 8]}
    #
    # grid_search =GridSearchCV(DecisionTreeClassifier(), param_grid=tuned_parameters, cv=5, scoring='accuracy')
    #
    # print("Performing grid search...")
    # print("pipeline: DecisionTreeClassifier")
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

    print "DecisionTreeClassifier"
    dtc = DecisionTreeClassifier(max_depth=10)
    acc_scores = cross_val_score(dtc,all_data_list, all_label_list, cv= 5, scoring='accuracy')
    print "acc_scores:",
    print acc_scores

    precision = cross_val_score(dtc,all_data_list, all_label_list, cv= 5, scoring='precision')
    print "precision:",
    print precision

    recall = cross_val_score(dtc,all_data_list, all_label_list, cv= 5, scoring='recall')
    print "recall:",
    print recall

    f1 = cross_val_score(dtc,all_data_list, all_label_list, cv= 5, scoring='f1')
    print "f1:",
    print f1

    roc_auc = cross_val_score(dtc,all_data_list, all_label_list, cv= 5, scoring='roc_auc')
    print "roc_auc:",
    print roc_auc

    print "mean of accuracy: %.6f, precision:%.6f, recall:%.6f, f1:%.6f, roc_auc:%.6f" % (acc_scores.mean(), precision.mean(), recall.mean(), f1.mean(), roc_auc.mean())
