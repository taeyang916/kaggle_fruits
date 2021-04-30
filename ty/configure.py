
# model edit
cfg_model = {'personal_net': [32, 32, 'M', 64, 64, 128, 128, 128, 'M', 256, 256, 256, 512, 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class Configure():
    def make_configure(cfg_type):
        cfg = Configure.set_cfg(cfg_type)
        return cfg

    def set_cfg(cfg_type):
        cfg = cfg_model['personal_net'] if cfg_type == 'personal_net' else cfg_model('vgg16')
        return cfg