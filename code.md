# model.fit(all_train_data_list, all_train_label_list,
    #               batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[checkpointer
    #         , lrate])


    # out_params = {
    #               "out_data_length" : sub_total,
    #               "n_time_steps" : n_time_steps,
    #               "data_dim" : data_dim,
    #               "time_interval" : time_interval,
    #               "spatial_interval" : spatial_interval,
    #               "n_lng": width,
    #               "n_lat": height
    #           }
    # data_dim = out_params["data_dim"]
    # timesteps = out_params["n_time_steps"]
    #
    # input_sequences = Input(shape=(timesteps, data_dim))
    #
    # #2 layer lstm
    # # for lstm_dim in lstm_dims:
    # lstm_dim = n_time_steps
    #
    # lstm1 = LSTM(lstm_dim, return_sequences=True)(input_sequences)
    # lstm1 = Dropout(0.3)(lstm1)
    # lstm2 = LSTM(lstm_dim)(lstm1)
    # lstm2 = Dropout(0.3)(lstm2)
    #
    #

    #
    #
    #
    #     print "start modeling"
    #     x = Dense(128, activation='relu')(lstm2)
    #     x = Dropout(0.4)(x)
    #     x = Dense(128, activation='relu')(x)
    #     x = Dropout(0.4)(x)
    #     main_output = Dense(3, activation='softmax', name='main_output')(x)
    #     #model = load_model("ckpt_pure_lstm_"+str(i_size)+".h5")
    #     model = Model(inputs=input_sequences, outputs=main_output)
    #     # model.compile(loss='binary_crossentropy', #loss :rmse?
    #     #               optimizer='rmsprop',# optimizer: adam?
    #     #               metrics=['accuracy'])
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
    #
    #     epochs = int(math.ceil(float(len(all_train_label_list))/float(steps_per_epoch * batch_size)))
    #
    #     validation_steps = int(math.ceil(float(len(all_val_label_list))/float(batch_size)))
    #     print "epochs %d" % epochs
    #
    #     print "start fitting model"
    #
    #     model.fit_generator(generate_arrays_of_train(all_train_data_list, all_train_label_list, batch_size),
    #     steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_arrays_of_validation(all_val_data_list, all_val_label_list, batch_size), validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate])#, initial_epoch=28)
    #
    #
    # print "size of all_train_label_list %d" % len(all_train_label_list)




    # LSTM_dim = 64
    # region_dim = 12
    # dense_dim = 64
    # validate_data_ratio = 1.0 - train_data_ratio
    # Input tensor for sequences of 20 timesteps,
    # each containing a 784-dimensional vector
    # lstm_layers = [1, 2, 3]
    # lstm_dims = [128, 64, 32, 16]
    # batch_sizes = [64, 128, 256]
    # dropouts = [0.2, 0.4, 0.6]
    # dense_layers = [0,1,2]

    #

        # model.fit_generator(generate_function_arrays_of_train(all_train_data_list, all_train_label_list, all_train_function_list, batch_size),
        # steps_per_epoch = steps_per_epoch, epochs=epochs, validation_data=generate_function_arrays_of_validation(all_val_data_list, all_val_label_list, all_val_function_list, batch_size), validation_steps = validation_steps, max_q_size=500,verbose=1,nb_worker=1, callbacks=[checkpointer, lrate,early_stoping])

        # model.fit(all_train_data_list, all_train_label_list,
        #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[checkpointer, lrate,early_stoping])




        # model_2_layer = Model(inputs=input_sequences, outputs=main_output)

        #
        # print "start fitting model with batch_size: %d" % batch_size
        # model.fit([all_train_data_list, all_train_function_list], all_train_label_list,
        #           batch_size=batch_size, epochs= epochs,validation_data=([all_val_data_list, all_val_function_list],all_val_label_list), callbacks=[lrate, early_stoping,checkpointer])


    # model_2_layer.compile(loss='mse', #loss :rmse?
    #               optimizer='adam',# optimizer: adam?
    #               metrics=['accuracy'])
    #
    # print "start fitting model_2_layer with adam"
    # model_2_layer.fit(all_train_data_list, all_train_label_list,
    #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[])
    # x = Dense(128, activation='relu')(lstm)
    # x = Dropout(0.2)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # main_output_dense = Dense(1, activation='sigmoid', name='main_output')(x)
    # model_2_layer_with_dense = Model(inputs=input_sequences, outputs=main_output_dense)
    # model_2_layer_with_dense.compile(loss='mse', #loss :rmse?
    #               optimizer='rmsprop',# optimizer: adam?
    #               metrics=['accuracy'])
    #
    # print "start fitting model_2_layer with dense"
    # model_2_layer_with_dense.fit(all_train_data_list, all_train_label_list,
    #           batch_size=batch_size, epochs= epochs,validation_data=(all_val_data_list,all_val_label_list), callbacks=[])
    #
    #

    # model = load_model('ckpt_pure_lstm.h5')
    #
    # score, acc = model.evaluate([all_val_data_list, all_val_function_list],all_val_label_list,
    #                             batch_size=batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)