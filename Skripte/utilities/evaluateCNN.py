import os
import numpy as np
import sys
import utilities.data_handling as datahandler
from CNN.CNN_classify import classfiy
import utilities.plot as plotlib


def evaluateCNN(config):

    # change logspath to initialLogpath/'eval_modelpath'_'test_set_path'
    config.logspath = config.eval_modelpath[:-1] + '_' + config.test_set.split('/')[-2] + '/'

    if not os.path.isdir(config.logspath) and (config.plot or config.write_file):
        os.makedirs(config.logspath)  # make sure directory exists for writing the file

    if 'etrend' in config.eval_modelpath:
        config.preprocessing = 'detrend'
    elif 'FFT' in config.eval_modelpath:
        config.preprocessing = 'fourier_channels'
    elif 'STFT' in config.eval_modelpath:
        config.preprocessing = 'stft'

    # get data info
    seq_length, n_channels, sensors = datahandler.read_infotxt(config.test_set)

    testlabels = datahandler.read_file('_labels', config.test_set, file_format='csv')
    testdata = datahandler.read_file('_data', config.test_set, file_format='csv')

    if testlabels.ndim == 1:
        testlabels, _ = datahandler.make_one_hot(testlabels)

    # reduce channels if specified
    if config.sel_sensors != sensors:
        testdata, sensors, n_channels = datahandler.select_channels(testdata, sensors, seq_length, config.sel_sensors)
    testdata, _, _ = datahandler.do_preprocessing(testdata, config.preprocessing, n_channels, seq_length)

    # transform data with scaler
    if config.preprocessing is 'fourier_samples':
        toscale = 'samples'
    else:
        toscale = 'channels'
    scaler = datahandler.load_scaler(config.eval_modelpath)
    testdata = datahandler.do_scaling(testdata, scaler, n_channels, seq_length, toscale=toscale)

    eval_data = datahandler.make_datasets(None, None, datahandler.make_dataset(testdata, testlabels))

    subDirectories = next(os.walk(config.eval_modelpath))[1]

    baselogspath = config.logspath
    baseevalmodelpath = config.eval_modelpath



    for subDirectory in subDirectories:

        run_for_loop_subdirectory(config=config, subDirectory=subDirectory, baseevalmodelpath=baseevalmodelpath,
                                  baselogspath=baselogspath, eval_data=eval_data, testlabels=testlabels)


def run_for_loop_subdirectory(config=None, subDirectory=None, baseevalmodelpath=None, baselogspath=None,
                              eval_data=None, testlabels=None):

    currentEvalPath = os.path.normpath(os.path.join(baseevalmodelpath, subDirectory))
    currentLogsPath = os.path.normpath(os.path.join(baselogspath, subDirectory))

    if os.path.isfile(currentEvalPath + os.path.sep + 'fold_0' + os.path.sep + 'metagraph.meta'):
        results = []
        foldDirectories = next(os.walk(currentEvalPath))[1]
        for foldDirectory in foldDirectories:
            config.eval_modelpath = os.path.normpath(os.path.join(currentEvalPath, foldDirectory)) + os.path.sep
            config.logspath = os.path.normpath(os.path.join(currentLogsPath, foldDirectory)) + os.path.sep
            if config.write_file:
                if not os.path.isdir(config.logspath) and (config.plot or config.write_file):
                    os.makedirs(config.logspath)  # make sure directory exists for writing the file
                orig_stdout = sys.stdout  # store original stdout
                f = open(config.logspath + 'log.txt', 'w+')
                sys.stdout = f
            result, confmat = classfiy(config, eval_data)
            results.append(result)
            if config.plot:
                plotlib.plot_confusionmatrix(confmat, config.logspath, [0, 100, 101, 104], export=config.export)
            if config.write_file:
                sys.stdout = orig_stdout
                np.savetxt(config.logspath + 'TrueLabels.csv', testlabels, delimiter=",")
        # write results of all folds in one file
        if config.write_file:
            cvlog = open(os.path.normpath(os.path.join(baselogspath, subDirectory)) + '/CV_log.txt', 'w+')
            cvlog.write("Accuracy: %s\n" % results)
            cvlog.close()

    else:

        subsubDirectories = next(os.walk(currentEvalPath))[1]

        for subsubDirectory in subsubDirectories:
            run_for_loop_subdirectory(config=config, subDirectory=subsubDirectory,
                                      baseevalmodelpath=currentEvalPath, baselogspath=currentLogsPath,
                                      eval_data=eval_data, testlabels=testlabels)

