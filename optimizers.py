import configparser
default_opt_cfg = 'optimizers.cfg'


def writeDefaultConfig():
    'Write default values to optimizer configuration file'
    cfg = configparser.ConfigParser()
    cfg['sgd'] = {'lr': '0.01', 'momentum': '0.0', 'decay': '0.0', 'nesterov': 'False'}
    cfg['rmsprop'] = {'lr': '0.001', 'rho': '0.9', 'epsilon': 'None', 'decay': '0.0'}
    cfg['adagrad'] = {'lr': '0.01', 'epsilon': 'None', 'decay': '0.0'}
    cfg['adadelta'] = {'lr': '1.0', 'rho': '0.95', 'epsilon': 'None', 'decay': '0.0'}
    cfg['adam'] = {'lr': '0.001', 'beta_1': '0.9', 'beta_2': '0.999', 'epsilon': 'None', 'decay': '0.0', 'amsgrad': 'False'}
    cfg['adamax'] = {'lr': '0.002', 'beta_1': '0.9', 'beta_2': '0.999', 'epsilon': 'None', 'decay': '0'}
    cfg['nadam'] = {'lr': '0.002', 'beta_1': '0.9', 'beta_2': '0.999', 'epsilon': 'None', 'schedule_decay': '0.004'}
    with open(default_opt_cfg, 'w') as f:
        cfg.write(f)


def parseParam(target_optimizer: str, param: str, opt_cfg: configparser.ConfigParser):
    'Parse and return parameter value'
    a = opt_cfg[target_optimizer][param].upper()
    if a == 'NONE':
        return None
    if a == 'FALSE':
        return False
    if a == 'TRUE':
        return True
    return float(a)


def getOptimizer(opt: str):
    'Returns a Keras optimizer object'
    import os
    from keras import optimizers as k_opt

    if not os.path.isfile(default_opt_cfg):
        writeDefaultConfig()
    opt_cfg = configparser.ConfigParser()
    opt_cfg.read(default_opt_cfg)
    if opt == 'sgd':
        return k_opt.SGD(lr=parseParam(opt, 'lr', opt_cfg), momentum=parseParam(opt, 'momentum', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg), nesterov=parseParam(opt, 'nesterov', opt_cfg))
    if opt == 'rmsprop':
        return k_opt.RMSprop(lr=parseParam(opt, 'lr', opt_cfg), rho=parseParam(opt, 'rho', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg))
    if opt == 'adagrad':
        return k_opt.Adagrad(lr=parseParam(opt, 'lr', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg))
    if opt == 'adadelta':
        return k_opt.Adadelta(lr=parseParam(opt, 'lr', opt_cfg), rho=parseParam(opt, 'rho', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg))
    if opt == 'adam':
        return k_opt.Adam(lr=parseParam(opt, 'lr', opt_cfg), beta_1=parseParam(opt, 'beta_1', opt_cfg), beta_2=parseParam(opt, 'beta_2', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg), amsgrad=parseParam(opt, 'amsgrad', opt_cfg))
    if opt == 'adamax':
        return k_opt.Adamax(lr=parseParam(opt, 'lr', opt_cfg), beta_1=parseParam(opt, 'beta_1', opt_cfg), beta_2=parseParam(opt, 'beta_2', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), decay=parseParam(opt, 'decay', opt_cfg))
    if opt == 'nadam':
        return k_opt.Nadam(lr=parseParam(opt, 'lr', opt_cfg), beta_1=parseParam(opt, 'beta_1', opt_cfg), beta_2=parseParam(opt, 'beta_2', opt_cfg), epsilon=parseParam(opt, 'epsilon', opt_cfg), schedule_decay=parseParam(opt, 'schedule_decay', opt_cfg))
    raise ValueError('Unrecognized optimizer ({})'.format(opt))
