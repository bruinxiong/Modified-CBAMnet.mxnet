import mxnet as mx
import symbol_utils
# Coded by Lin Xiong on Sep-25, 2018
# Referred to the pytorch code https://github.com/Youngkl0726/Convolutional-Block-Attention-Module/blob/master/CBAMNet.py, 
# More detailed information can be found in the following paper:
# Sanghyun Woo, Jongchan Park, Joon-Young Lee and In So Kweon, "CBAM: Convolutional Block Attention Module", ECCV 2018, https://arxiv.org/pdf/1807.06521v2.pdf
# We also refer the input setting and block setting of this paper II:
# Jiankang Deng, Jia Guo and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.o7698v1
# The size of input faces is only 112x112 not 224x224
# We also refer the resnet-v2 version proposed in the following paper:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"

bn_mom = 0.9

# Basic layers
def BN(data, momentum=bn_mom, fix_gamma=False, eps=2e-5, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, eps=eps, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, eps=eps, momentum=momentum)
    return bn

def Act(data, act_type='prelu', name=None):
    body = mx.sym.LeakyReLU(data = data, act_type=act_type, name = '%s_%s' %(name, act_type))
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=256, name=None, w=None, b=None, suffix=''):
    if w is None:
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, workspace=workspace, name='%s%s_conv2d' %(name, suffix))
    else:
        if b is None:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, workspace=workspace, weight=w, name='%s%s_conv2d' %(name, suffix))
        else:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, workspace=workspace, weight=w, bias=b, name='%s%s_conv2d' %(name, suffix))
    return conv

def BN_Act(data, momentum=bn_mom, name=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_act = Act(bn, act_type='prelu', name=name)
    return bn_act

def BN_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, workspace=256, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_conv = Conv(bn, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, workspace=workspace, name=name, w=w, b=b, suffix=suffix)
    return bn_conv

def BN_Act_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, workspace=256, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_act = Act(bn, act_type='prelu', name=name)
    bn_act_conv = Conv(bn_act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, workspace=workspace, name=name, w=w, b=b, suffix=suffix)
    return bn_act_conv

# Convolutional Block Attention Module (CBAM)
def CBAM(data, num_filter, reduction, act_type, workspace, name, suffix=''):
    # Channel attention module
    module_input = data
    avg = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='avg', name='%s_ca_avg_pool1' %(name))
    ma  = mx.sym.Pooling(data=data, global_pool=True, kernel=(7, 7), pool_type='max', name='%s_ca_max_pool1' %(name))
    # import pdb
    # pdb.set_trace()
    avg = Conv(avg, num_filter=num_filter//reduction, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_ca_avg_fc1' %(name), suffix='')
    ma  = Conv(ma, num_filter=num_filter//reduction, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_ca_max_fc1' %(name), suffix='')
    # import pdb
    # pdb.set_trace()
    avg = Act(avg, act_type=act_type, name='%s_ca_avg_%s' %(name, act_type))
    ma  = Act(ma, act_type=act_type, name='%s_ca_max_%s' %(name, act_type))
    avg = Conv(avg, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_ca_avg_fc2' %(name), suffix='')
    ma  = Conv(ma, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_ca_max_fc2' %(name), suffix='')
    # import pdb
    # pdb.set_trace()
    body = avg + ma
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_ca_sigmoid' %(name))
    # Spatial attention module
    body = mx.symbol.broadcast_mul(module_input, body)
    # import pdb
    # pdb.set_trace()
    module_input = body
    avg = mx.symbol.mean(data=body, axis=1, keepdims=True, name='%s_sa_mean' %(name))
    ma  = mx.symbol.max(data=body, axis=1, keepdims=True, name='%s_sa_max' %(name))
    # import pdb
    # pdb.set_trace()
    body = mx.symbol.Concat(avg, ma, dim=1, name='%s_sa_concat' %(name))
    body = Conv(body, num_filter=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, workspace=workspace, name='%s_sa_conv' %(name), suffix='')
    body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_sa_sigmoid' %(name))
    body = mx.symbol.broadcast_mul(module_input, body)
    return body

# Instance-batch normalization (IBN) block
def IBN_block(data, num_filter, name, eps=2e-5, bn_mom=0.9, suffix=''):
    split = mx.symbol.split(data=data, axis=1, num_outputs=2)
    # import pdb
    # pdb.set_trace()
    out1 = mx.symbol.InstanceNorm(data=split[0], eps=eps, name=name + '_ibn' + '_in1')
    out2 = BN(split[1], momentum=bn_mom, fix_gamma=False, eps=eps, name=name + '_ibn', suffix=suffix)
    out = mx.symbol.Concat(out1, out2, dim=1, name=name + '_ibn1')
    return out

def CBAM_Residual_unit(data, num_filter, reduction, stride, dim_match, name, bottle_neck, **kwargs):
    # Improved resnet bottleneck with a CBAM module. It follows the paper "ArcFace: Additive Angular Margin Loss for Deep Face Recognition". 
    # We also refer the paper "Identity Mappings in Deep Residual Networks".
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    reduction = kwargs.get('reduction', 16)
    act_type = kwargs.get('version_act', 'prelu')
    ibn = kwargs.get('ibn', False)
    memonger = kwargs.get('memonger', False)

    if bottle_neck:
        if num_filter == 2048:
          ibn = False
        if ibn:
          bn1 = IBN_block(data=data, num_filter=int(num_filter*0.25), name='%s_c1x1' %(name))
        else:
          bn1 = BN(data, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_c1x1' %(name), suffix='')
        conv1 = Conv(bn1, num_filter=int(num_filter*0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_c1x1_a' %(name), suffix='')
        conv2 = BN_Act_Conv(conv1, num_filter=int(num_filter*0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_c3x3' %(name))
        conv3 = BN_Act_Conv(conv2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1,
                                         momentum=bn_mom, workspace=workspace, name='%s_c1x1_b' %(name))
        conv3 = BN(conv3, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_c1x1_b' %(name))
        # import pdb
        # pdb.set_trace()
        conv3 = CBAM(conv3, num_filter, reduction, act_type, workspace, name='%s_cbam' %(name))
        # import pdb
        # pdb.set_trace()
        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, workspace=workspace, name='%s_conv1sc' %(name), suffix='')
            shortcut = BN(conv1sc, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_sc' %(name))
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        if num_filter == 512:
          ibn = False
        if ibn:
          bn1 = IBN_block(data=data, num_filter=num_filter, name='%s_c3x3' %(name))
        else:
          bn1 = BN(data, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_c3x3' %(name), suffix='')
        conv1 = Conv(bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, workspace=workspace, name='%s_c3x3_a' %(name), suffix='')
        conv2 = BN_Act_Conv(conv1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_c3x3_b' %(name))
        conv2 = BN(conv2, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_c3x3_b' %(name))
        # import pdb
        # pdb.set_trace()
        conv2 = CBAM(conv2, num_filter, reduction, act_type, workspace, name='%s_cbam' %(name))
        # import pdb
        # pdb.set_trace()
        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, workspace=workspace, name='%s_conv1sc' %(name), suffix='')
            shortcut = BN(conv1sc, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_sc' %(name))
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def CBAMNet(units, num_stages, filter_list, num_classes, bottle_neck, **kwargs):
    global bn_mom
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    bn_mom = kwargs.get('bn_mom', 0.9)
    input_shape = kwargs.get('input_shape', None)
    reduction = kwargs.get('reduction', 16)
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    act_type = kwargs.get('version_act', 'prelu')
    print(version_input, version_output, act_type)
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data', shape=input_shape)
    data = mx.sym.identity(data=data, name='id')
    data = data-127.5
    data = data*0.0078125
    if version_input==0:
      body = Conv(data, filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), num_group=1, workspace=workspace, name='cbam_conv1')
      body = BN_Act(body, momentum=bn_mom, name='cbam_conv1_bn1')
      body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1,1), pool_type='max')
    else:
      body = Conv(data, filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, workspace=workspace, name='cbam_conv1')
      body = BN_Act(body, momentum=bn_mom, name='cbam_conv1_bn1')
    body._set_attr(mirror_stage='True')

    for i in range(num_stages):
      if version_input==0:
        body = CBAM_Residual_unit(body, filter_list[i+1], reduction, (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      else:
        body = CBAM_Residual_unit(body, filter_list[i+1], reduction, (2, 2), False,
          name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
      # import pdb
      # pdb.set_trace()
      body._set_attr(mirror_stage='True')
      for j in range(units[i]-1):
        body = CBAM_Residual_unit(body, filter_list[i+1], reduction, (1, 1), True, name='stage%d_unit%d' % (i+1, j+2),
          bottle_neck=bottle_neck, **kwargs)
        # import pdb
        # pdb.set_trace()
        body._set_attr(mirror_stage='True')

    # import pdb
    # pdb.set_trace()
    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    fc1._set_attr(mirror_stage='True')
    return fc1

def get_symbol(num_classes, num_layers, **kwargs):

    if num_layers >= 101:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return CBAMNet(units      = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  **kwargs)

