from keras.models import Model


def extended_summary(model: Model):
    cfg = model.get_config()
    fields = ('#', 'Name', 'Type', 'Activation', 'Trainable', 'Filters',
              'Kernel size', 'Strides', 'Padding', 'Pool Size', 'Units')
    print('%-10s%-30s%-30s%-20s%-15s%-15s%-15s%-15s%-15s%-15s%-15s' % fields)
    i = 0
    for l in cfg['layers']:
        values = (i, l['name'], l['class_name'],
                  l['config']['activation'] if 'activation' in l['config'] else '-',
                  l['config']['trainable'] if 'trainable' in l['config'] else '-',
                  l['config']['filters'] if 'filters' in l['config'] else '-',
                  l['config']['kernel_size'] if 'kernel_size' in l['config'] else '-',
                  l['config']['strides'] if 'strides' in l['config'] else '-',
                  l['config']['padding'] if 'padding' in l['config'] else '-',
                  l['config']['pool_size'] if 'pool_size' in l['config'] else '-',
                  l['config']['units'] if 'units' in l['config'] else '-'
                  )
        print('%-10i%-30s%-30s%-20s%-15s%-15s%-15s%-15s%-15s%-15s%-15s' % values)
        i += 1