#!/usr/bin/env python
def paramDictFormat(d: dict):
    'Formats dictionary into a comma-separated key=value style string'
    return ', '.join([f'{key}={val:.5f}' if isinstance(val, float) else f'{key}={val}' for key, val in d.items()])


def main():
    'Main'
    from argparse import ArgumentParser
    import os

    # available architectures
    models_list = [
        'vgg16',
        'vgg19',
        'inceptionv3',
        'resnet50',
        'custom',
        'xception',
        'inceptionresnet',
        'mobilenet',
        'densenet121',
        'densenet169',
        'densenet201',
        'nasnet',
        'mobilenetv2'
    ]

    # available optimizers
    optimizers_list = ['sgd', 'adam']

    losses_list = [
        'categorical_crossentropy',
        'sparse_categorical_crossentropy',
        'binary_crossentropy',
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'squared_hinge',
        'hinge',
        'categorical_hinge',
        'logcosh',
        'kullback_leibler_divergence',
        'poisson',
        'cosine_proximity'
    ]

    # print these names for loss functions
    losses_dict = {
        'mean_squared_error': 'MSE',
        'mean_absolute_error': 'MAE',
        'mean_absolute_percentage_error': 'MAPE',
        'mean_squared_logarithmic_error': 'MSLE',
        'squared_hinge': 'Squared Hinge',
        'hinge': 'Hinge',
        'categorical_hinge': 'Categorical Hinge',
        'logcosh': 'Log-Cosh',
        'categorical_crossentropy': 'Categorial Cross-entropy',
        'sparse_categorical_crossentropy': 'Sparse Categorical Cross-entropy',
        'binary_crossentropy': 'Binary Cross-entropy',
        'kullback_leibler_divergence': 'Kullback-Leibler Divergence',
        'poisson': 'Poisson',
        'cosine_proximity': 'Cosine Proximity'
    }

    parser = ArgumentParser()
    parser.add_argument('model', help='which model to use',
                        type=str, choices=models_list)
    parser.add_argument('path', help='path to data', type=str)
    parser.add_argument('--loadfrom', help='load previous model', type=str)
    parser.add_argument(
        '-e', '--epochs', help='epochs to train for', type=int, default=30)
    parser.add_argument(
        '-b', '--batch', help='training batch size', type=int, default=8)
    parser.add_argument('-o', '--optimizer', help='optimizer to use',
                        type=str, choices=optimizers_list, default='sgd')
    parser.add_argument(
        '-s', '--split', help='test split size', default=0.2, type=float)
    parser.add_argument('-t', '--testset', help='path to test data', type=str)
    parser.add_argument('--loss', help='loss function to use', type=str,
                        choices=losses_list, default='categorical_crossentropy')
    parser.add_argument('--nogpu', help='disable GPU',
                        action='store_true', dest='no_gpu')
    parser.add_argument('--usemp', help='enable multiprocessing for sequences',
                        action='store_true', dest='use_mp')
    parser.add_argument(
        '--pretrained', help='load pre-trained weights', action='store_true')
    parser.add_argument('--output', help='output file', type=str)
    parser.add_argument('--extsum', help='print extended summary',
                        action='store_true', dest='print_extsum')
    parser.add_argument('--sum', help='print summary',
                        action='store_true', dest='print_sum')
    parser.add_argument('--json', help='save model as JSON file', type=str)
    parser.add_argument('--log', help='test log filename',
                        type=str, default='tests_log.log')
    parser.add_argument(
        '--shape', help='input shape in (height:width:depth) format', type=str)
    parser.add_argument(
        '--dropout', help='dropout probability (default=0)', type=float)
    parser.add_argument('--pfldir', help='put .pfl files here',
                        type=str, default='.', dest='pfldir')
    parser.add_argument('--decay', help='weight decay',
                        default=0.005, type=float, dest='weight_decay')
    parser.add_argument('-K', help='apply kernel regularization',
                        action='store_true', dest='regularize_kernel')
    parser.add_argument('-B', help='apply bias regularization',
                        action='store_true', dest='regularize_bias')
    parser.add_argument('-A', help='apply activity regularization',
                        action='store_true', dest='regularize_activity')
    parser.add_argument(
        '-a', '--augment', help='apply perform data augmentation', action='store_true')
    parser.add_argument(
        '--seed', help='random seed for train-test split', type=int, default=7)
    args = parser.parse_args()

    from keras.callbacks import Callback

    if args.pfldir:
        os.makedirs(args.pfldir, exist_ok=True)
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    class PerformanceHistory(Callback):
        def __init__(self, output_file: str):
            super().__init__()
            self.output_file = open(output_file, 'wt')
            self.csv_file = csv.writer(self.output_file)
            self.csv_file.writerow(
                ['ACCURACY', 'LOSS', 'VALIDATION ACCURACY', 'VALIDATION LOSS'])

        def on_epoch_end(self, batch, logs={}):
            self.csv_file.writerow([logs.get('accuracy'), logs.get(
                'loss'), logs.get('val_accuracy'), logs.get('val_loss')])
            self.output_file.flush()

        def __del__(self):
            self.output_file.close()

    if args.no_gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    image_dim = 224 if args.model in ['vgg16', 'vgg19', 'custom'] else 299
    input_shape = (image_dim, image_dim, 3)
    if args.shape is not None:
        input_shape = [(int(y) if y != '' else 1)
                       for y in args.shape.split(':')]

    import keras.applications as apps

    # preprocessing functions dictonary
    input_preprocessing = {
        'vgg16': apps.vgg16.preprocess_input,
        'vgg19': apps.vgg19.preprocess_input,
        'inceptionv3': apps.inception_v3.preprocess_input,
        'resnet50': apps.resnet50.preprocess_input,
        'custom': apps.vgg16.preprocess_input,
        'xception': apps.xception.preprocess_input,
        'inceptionresnet': apps.inception_resnet_v2.preprocess_input,
        'mobilenet': apps.mobilenet.preprocess_input,
        'densenet121': apps.densenet.preprocess_input,
        'densenet169': apps.densenet.preprocess_input,
        'densenet201': apps.densenet.preprocess_input,
        'nasnet': apps.nasnet.preprocess_input,
        'mobilenetv2': apps.mobilenet_v2.preprocess_input
    }

    from keras.layers import Dropout
    from keras.regularizers import l2
    from keras.models import load_model
    from keras import Model
    from optimizers import getOptimizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer
    from generator import AugmentedBatchGenerator, BatchGenerator
    from extsum import extended_summary
    import numpy as np
    import datetime as dt
    import pickle
    import csv
    import subprocess
    from random import shuffle, seed

    # load default specified dataset
    data_path = os.path.abspath(args.path)
    X_raw = []
    y_raw = []

    for d in os.listdir(data_path):
        for f in os.listdir(data_path + '/' + d):
            X_raw.append('/'.join([data_path, d, f]))
            y_raw.append(d)

    lb = LabelBinarizer()
    lb.fit(y_raw)
    seed(args.seed)

    if args.testset:
        # if a test set was specified, load it
        print(f'Using data at {args.testset} as test set.')

        # shuffle training data
        training_shuffled = list(zip(X_raw, y_raw))
        shuffle(training_shuffled)
        X_data, y_data = zip(*training_shuffled)

        X_data, y_data = np.asarray(X_data), lb.transform(y_data)
        X_train, y_train = X_data, y_data
        data_path = os.path.abspath(args.testset)

        X_raw = []
        y_raw = []

        for d in os.listdir(data_path):
            for f in os.listdir(data_path + '/' + d):
                X_raw.append('/'.join([data_path, d, f]))
                y_raw.append(d)

        # shuffle test
        test_shuffled = list(zip(X_raw, y_raw))
        shuffle(test_shuffled)
        X_raw, y_raw = zip(*test_shuffled)

        X_test, y_test = np.asarray(X_raw), lb.transform(y_raw)
    else:
        # otherwise split default dataset
        print(f'Using {args.split} of the provided dataset as test set.')

        X_data, y_data = np.asarray(X_raw), lb.transform(y_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=args.split, random_state=args.seed)

    del X_raw
    del y_raw

    n_classes = y_data.shape[1]

    models_dict = {
        'xception': apps.Xception,
        'vgg16': apps.VGG16,
        'vgg19': apps.VGG19,
        'resnet50': apps.ResNet50,
        'inceptionv3': apps.InceptionV3,
        'inceptionresnet': apps.InceptionResNetV2,
        'mobilenet': apps.MobileNet,
        'mobilenetv2': apps.MobileNetV2,
        'densenet121': apps.DenseNet121,
        'densenet169': apps.DenseNet169,
        'densenet201': apps.DenseNet201,
        'nasnet': apps.NASNetLarge
    }

    # load vanilla model with specified parameters
    model = models_dict[args.model](
        classes=n_classes, input_shape=input_shape, weights='imagenet' if args.pretrained else None)

    if args.dropout is not None:
        print('Adding weight decay')
        # insert dropout layer and regularization
        preds = model.layers[-1]
        dp = Dropout(args.dropout)(model.layers[-2].output)
        preds = preds(dp)
        model = Model(inputs=model.inputs, outputs=preds)

        for layer in model.layers:
            if args.regularize_kernel:
                layer.kernel_regularizer = l2(args.weight_decay)
            if args.regularize_bias:
                layer.bias_regularizer = l2(args.weight_decay)
            if args.regularize_activity:
                layer.activity_regularizer = l2(args.weight_decay)

    opt = getOptimizer(args.optimizer)

    model.compile(loss=args.loss, optimizer=opt, metrics=['accuracy'])
    if args.loadfrom:
        print('Loading', args.loadfrom)
        model = load_model(os.path.abspath(args.loadfrom))

    # iteratively rename performance file
    pfldir = os.path.abspath(args.pfldir)
    performance_file = os.path.join(
        pfldir, f'{args.model}_b{args.batch}_e{args.epochs}.pfl')
    fnum = 1
    while os.path.isfile(performance_file):
        performance_file = os.path.join(
            pfldir, f'{args.model}_b{args.batch}_e{args.epochs}_{fnum}.pfl')
        fnum += 1
    os.makedirs(pfldir, exist_ok=True)

    if args.print_extsum:
        extended_summary(model)
    elif args.print_sum:
        model.summary()

    perf_log = PerformanceHistory(performance_file)
    # print test parameters to screen
    print('\n{:<20}{}'.format('Model', args.model))
    print('{:<20}{}'.format('Input shape', input_shape))
    print('{:<20}{}'.format('Epochs', args.epochs))
    print('{:<20}{}'.format('Batch size', args.batch))
    print('{:<20}{}'.format('Optimizer', type(opt).__name__))
    print('{:<20}{}'.format('Optimizer params',
                            paramDictFormat(opt.get_config())))
    print('{:<20}{}'.format('Loss', args.loss))
    print('{:<20}{}'.format('Multiprocessing', 'On' if args.use_mp else 'Off'))
    print('{:<20}{}'.format('Performance log', performance_file))
    print('{:<20}{}'.format('Test log', args.log))
    print('{:<20}{}'.format('Dataset', args.path))
    reg = []
    if args.regularize_kernel:
        reg.append('kernel')
    if args.regularize_activity:
        reg.append('activity')
    if args.regularize_bias:
        reg.append('bias')
    print('{:<20}{}\n'.format('Regularization',
                              'None' if not reg else ', '.join(reg)))

    opt = getOptimizer(args.optimizer)
    model.compile(loss=args.loss, optimizer=opt, metrics=['accuracy'])

    # create training batch generator
    if args.augment:
        print('Data augmentation enabled.')
        train_gen = AugmentedBatchGenerator(X_train, y_train, args.batch, shape=input_shape, ops=[
                                            input_preprocessing[args.model]], pad=False)
    else:
        print('Data augmentation disabled.')
        train_gen = BatchGenerator(X_train, y_train, args.batch, shape=input_shape, ops=[
                                   input_preprocessing[args.model]], pad=False)
    # create testing batch  generator
    test_gen = BatchGenerator(X_test, y_test, args.batch, shape=input_shape, ops=[
                              input_preprocessing[args.model]], pad=False)

    # train model
    train_start = dt.datetime.now()
    model.fit_generator(train_gen, epochs=args.epochs, use_multiprocessing=args.use_mp,
                        validation_data=test_gen, callbacks=[perf_log])
    train_end = dt.datetime.now()

    # evaluate final model on train set
    train_score = model.evaluate_generator(train_gen)
    print('Train loss:', train_score[0])
    print('Train accuracy:', train_score[1])

    # evaluate final model on test set
    test_score = model.evaluate_generator(test_gen)
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])

    # update tests log with current test data
    date_format = '{:%Y-%m-%d %H:%M}'
    log_format = '{:<20}{:<20}{:<20}{:<10}{:<10}{:<15}{:<15.5}{:<15.5}{:<15.5}{:<15.5}{:<30}{:<30}{:<70}{:<15}{:<15}{:<15}{:<15.5}{:<15.5}\n'
    header_format = '{:<20}{:<20}{:<20}{:<10}{:<10}{:<15}{:<15}{:<15}{:<15}{:<15}{:<30}{:<30}{:<70}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
    with open(args.log, 'a+t') as test_log:
        if test_log.tell() <= 0:
            test_log.write(header_format.format(
                'BEGIN', 'END', 'ARCHITECTURE', 'BATCH', 'EPOCHS', 'OPTIMIZER', 'TRAIN LOSS', 'TRAIN ACC', 'TEST LOSS', 'TEST ACC', 'DATA FOLDER', 'LOSS FUNCTION', 'OPTIMIZER PARAMS',
                'KERNEL REG', 'BIAS REG', 'ACTIV. REG', 'DECAY', 'DROPOUT'))
        start_str = date_format.format(train_start)
        end_str = date_format.format(train_end)
        data_folder = args.path.split('/')[-1 if args.path[-1] != '/' else -2]

        test_log.write(log_format.format(start_str, end_str, args.model.upper(), args.batch, args.epochs, args.optimizer.upper(), train_score[0], train_score[1],
                                         test_score[0], test_score[1], data_folder, losses_dict[args.loss], paramDictFormat(
                                             opt.get_config()),
                                         'YES' if args.regularize_kernel else 'NO',
                                         'YES' if args.regularize_bias else 'NO',
                                         'YES' if args.regularize_activity else 'NO',
                                         args.weight_decay,
                                         args.dropout if args.dropout else 0.0))

    # save the model and class file if an output filename was specified
    if args.output is not None:
        print(f'Saving model as {args.output}.h5')
        os.makedirs('/'.join(args.output.split('/')[:-1]), exist_ok=True)
        model.save(f'{args.output}.h5')
        with open(f'{args.output}.bin', 'wb') as fout:
            pickle.dump((args.model, lb), fout)

    subprocess.run(['notify-send', 'Entrenamiento completado',
                    f'Se ha completado el entrenamiento del modelo {args.model}.'], check=False)


if __name__ == '__main__':
    main()
