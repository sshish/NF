########################################################################
#Basic interface of blockwise autoregressive monotonic transformations.#
########################################################################

import torch

import NF

class Basic(torch.nn.Module):
  """Abstract module for a blockwise autoregressive monotonic transformation.
    It is characterized by Jacobians of blockwise lower triangular form,
    whereby blocks result from channels being split im multiple features:

    +++ 0 0000
    +++ 0 0000

    ... + 0000
    ... + 0000
    ... + 0000

    ... . ++++

    This is an example of such a Jacobian, with input (and output) having 3
    channels. The 1st input channel has 3 features, the 2nd has 1 feature, the
    3rd has 4 features. For output, the features are (2,3,1). Diagonal blocks
    have positive entries (+), upper diagonal blocks are zero (0). BAM
    interface allows to create neural network implementations of flows like in
    the paper on "Block Neural Autoregressive Flows".
  Methods:
    forward(x, c): Similar to NF.forward(), except that the CHANNEL dimension
      is now the FEATURE dimension, and the second entry in the INFO dimension
      stores the value dy_j_i/dx_j, where the original channel j has been split
      into features j_1,j_2,...
    NF(): Wraps itself into a NF, if the number of input features and output
      features is 1 for all channels.

  """

  def __init__(self, *args):
    super(Basic, self).__init__()
    assert(isinstance(self._input_features, list))
    assert(isinstance(self._output_features, list))
    assert(len(self._input_features) == len(self._output_features))

  def NF(self):
    return FromBAM(self)

  def _get_name(self):
    return 'BAM:' + super(Basic, self)._get_name() + ' {} -> {}'.format(self._input_features, self._output_features)

class FromBAM(NF.Basic):
  """If the input and output have only 1 feature per channel, BAM can be
    converted to NF. The inverse() operation is performed via bisection, it
    is assumed that the true value x lies between -1 and 1 for all entries.

  """

  def __init__(self, net, minimum=None, maximum=None, max_iterations=10, tolerance=1e-3, randomize=False):
    assert(isinstance(net, Basic))
    assert(net._input_features == net._output_features)
    for feature in net._input_features:
      assert(feature == 1)
    if minimum is not None:
      assert(minimum.shape == torch.Size([len(net._input_features)]))
      self.bisection_minimum = minimum
    else: self.bisection_minimum = torch.empty(len(net._input_features)).fill_(-1.)
    if maximum is not None:
      assert(maximum.shape == torch.Size([len(net._input_features)]))
      self.bisection_maximum = maximum
    else: self.bisection_maximum = torch.empty(len(net._input_features)).fill_(1.)
    self.bisection_tolerance = tolerance
    self.bisection_max_iterations = max_iterations
    self.bisection_randomize = randomize
    super(FromBAM, self).__init__()
    self._net = net

  def forward(self, x, c):
    return self._net(x, c)

  def inverse(self, y, c):
    x = (0.5 * (self.bisection_minimum + self.bisection_maximum)).expand_as(y).to(y.device)
    x = torch.stack((x, torch.empty_like(x)), dim=2)
    for i in range(y.shape[1]):
      unsatisfied = torch.ones_like(y[:,0]).bool()
      _min = torch.ones_like(y[:,0]) * self.bisection_minimum[i]
      _max = torch.ones_like(y[:,0]) * self.bisection_maximum[i]
      iteration = -1
      while unsatisfied.any() and iteration < self.bisection_max_iterations:
        iteration += 1
        _y = self._net(x[unsatisfied], c[unsatisfied])[:,i,0]
        _min[unsatisfied] = _min[unsatisfied].where(_y > y[unsatisfied,i], x[unsatisfied,i,0])
        _max[unsatisfied] = _max[unsatisfied].where(_y < y[unsatisfied,i], x[unsatisfied,i,0])
        x[unsatisfied,i,0] = 0.5 * (_min + _max)[unsatisfied]
        unsatisfied[unsatisfied].where((_y - y[unsatisfied,i]).abs() > self.bisection_tolerance, torch.zeros_like(unsatisfied[unsatisfied]))
      if(self.bisection_randomize): x[:,i,0] = _min + torch.rand_like(_min) * (_max - _min)
    return x[:,:,0]

##########################################################################
#Implementations of different sublayers that comply to the BAM interface.#
##########################################################################

class Id(torch.nn.Sequential, Basic):
  """Identity transform.

  Args:
    features: List of feature sizes per input (output) channel.

  """

  def __init__(self, features):
    self._input_features = features
    self._output_features = features
    super(Id, self).__init__()

  def forward(self, x, _):
    return x

class Stack(torch.nn.Sequential, Basic):
  """A BAM that is created by stacking multiple BAMs.

  Args: Individual BAMs must be listed in the order from x-space (data) to
    y-space (latent variable).

  """

  def __init__(self, *args):
    for net in args:
      assert(isinstance(net, Basic))
    for i in range(len(args) - 1):
      assert(args[i]._output_features == args[i+1]._input_features)
    self._input_features = args[0]._input_features
    self._output_features = args[-1]._output_features
    super(Stack, self).__init__(*args)


  def forward(self, x, c):
    for net in self._modules.values():
      x = net(x, c)
    return x

class Cat(Basic):
  """Applies blockwise concatenation to the outputs of multiple BAMs.

  Args: Individual BAMs must have same input features.

  """

  def __init__(self, *args):
    for net in args:
      assert(isinstance(net, Basic))
    self._input_features = args[0]._input_features
    for net in args:
      assert(net._input_features == self._input_features)
    self._output_features = [sum(x) for x in zip(*[net._output_features for net in args])]
    super(Cat, self).__init__()
    for idx, net in enumerate(args):
      self.add_module(str(idx), net)

  def forward(self, x, c):
    splits = [list(net(x, c).split(net._output_features, dim=1)) for net in self._modules.values()]
    splits = [torch.cat(split, dim=1) for split in zip(*splits)]
    return torch.cat(splits, dim=1)

class Sum(Basic):
  """Applies blockwise summation over the outputs of a BAM.

  Args:
    in_features: List of feature sizes per input channel.

  """

  def __init__(self, in_features):
    self._input_features = in_features
    self._output_features = [1] * len(self._input_features)
    super(Sum, self).__init__()
    self._input_features_cum = [0]
    in_c = 0
    for i in range(len(self._input_features)):
      in_c += self._input_features[i]
      self._input_features_cum.append(in_c)

  def forward(self, x, _):
    result = torch.empty(x.shape[0],len(self._output_features),2, device=x.device, dtype=x.dtype)
    for i in range(len(self._input_features_cum) - 1):
      result[:,i,0] = x[:,self._input_features_cum[i]:self._input_features_cum[i+1],0].sum(dim=1)
      result[:,i,1] = x[:,self._input_features_cum[i]:self._input_features_cum[i+1],1].logsumexp(dim=1)
    return result

class Tanh(Basic):
  """Tanh layer with BAM interface (context-free).

  Args:
    features: List of feature sizes per input (output) channel.

  """

  def __init__(self, features):
    self._input_features = features
    self._output_features = features
    super(Tanh, self).__init__()

  def forward(self, x, _):
    _x = x[:,:,0]
    delta = - 2. * (_x - torch.empty_like(_x).fill_(2).log() + torch.nn.functional.softplus(- 2. * _x))
    return torch.stack((_x.tanh(), x[:,:,1] + delta), dim=2)

class Gate(Basic):
  """Context-free gating layer.

  Args:
    in_features: List of feature sizes per input channel.
    out_features: List of feature sizes per output channel.

  """

  def __init__(self, features):
    self._input_features = features
    self._output_features = features
    super(Gate, self).__init__()
    self._gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(sum(features))))

  def forward(self, x, _):
    delta = - (torch.nn.functional.softplus(-1. * self._gate))
    return torch.stack((x[:,:,0] * self._gate.sigmoid(), x[:,:,1] + delta), dim=2)

class CGate(Basic):
  """Context-dependent gating layer.

  """

  def __init__(self, features, gate):
    assert(isinstance(gate, torch.nn.Module))
    self._input_features = features
    self._output_features = features
    super(CGate, self).__init__()
    self._gate = gate

  def forward(self, x, c):
    gate = self._gate(c)
    delta = - (torch.nn.functional.softplus(-1. * gate))
    return torch.stack((x[:,:,0] * gate.sigmoid(), x[:,:,1] + delta), dim=2)

class Linear(Basic):
  """Context-free blockwise masked linear transformation with weight
    normalization. This is the basic building block from the paper on Block
    Neural Autoregressive Flows.

  Args:
    in_features: List of feature sizes per input channel.
    out_features: List of feature sizes per output channel.

  """

  def __init__(self, in_features, out_features):
    self._input_features = in_features
    self._output_features = out_features
    super(Linear, self).__init__()
    self._in_features_cum = [0]
    self._out_features_cum = [0]
    in_c = 0
    out_c = 0
    for i in range(len(in_features)):
      in_c += in_features[i]
      out_c += out_features[i]
      self._in_features_cum.append(in_c)
      self._out_features_cum.append(out_c)
    weight = torch.empty(self._out_features_cum[-1], self._in_features_cum[-1])
    for i in range(len(in_features)):
      weight[self._out_features_cum[i]:self._out_features_cum[i+1],0:self._in_features_cum[i+1]] = torch.nn.init.xavier_uniform_(torch.Tensor(out_features[i],self._in_features_cum[i+1]))
    self._weight_dir = torch.nn.Parameter(weight)
    self._bias = torch.nn.Parameter(torch.zeros(self._out_features_cum[-1],1))
    self._weight_amp = torch.nn.Parameter(torch.nn.init.uniform_(torch.Tensor(self._out_features_cum[-1],1), 0.5, 1.5).log())
    mask_d = torch.zeros(self._out_features_cum[-1], self._in_features_cum[-1]).bool()
    mask_o = torch.zeros(self._out_features_cum[-1], self._in_features_cum[-1]).bool()
    for i in range(len(in_features)):
      mask_d[self._out_features_cum[i]:self._out_features_cum[i+1],self._in_features_cum[i]:self._in_features_cum[i+1]]=1
      mask_o[self._out_features_cum[i]:self._out_features_cum[i+1],0:self._in_features_cum[i]]=1
    self.register_buffer("mask_d", mask_d)
    self.register_buffer("mask_o", mask_o)

  def forward(self, x, _):
    w = torch.zeros_like(self._weight_dir).where(~self.mask_d, self._weight_dir.exp()) + torch.zeros_like(self._weight_dir).where(~self.mask_o, self._weight_dir)
    squarednorm = (w ** 2).sum(dim=1, keepdim=True)
    w = w / squarednorm.sqrt()
    w = w * self._weight_amp.exp()
    _x = x[:,:,0,None]
    x_ = x[:,:,1,None]
    _x = (w @ _x) + self._bias
    logdiag = torch.empty_like(w).where(~self.mask_d, self._weight_dir - 0.5 * squarednorm.log() + self._weight_amp + x_.transpose(1,2).expand(-1,w.shape[0],-1))
    logdet = torch.zeros(_x.shape[0],0,1, device=_x.device)
    for i in range(len(self._input_features)):
      logdet = torch.cat((logdet, logdiag[:,self._out_features_cum[i]:self._out_features_cum[i+1],self._in_features_cum[i]:self._in_features_cum[i+1]].logsumexp(dim=2, keepdim=True)), dim=1)
    return torch.cat((_x, logdet), dim=2)

class CLinear(Basic):
  """Context-dependent blockwise masked linear transformation with weight
    normalization.

  Args:
    in_features: List of feature sizes per input channel.
    out_features: List of feature sizes per output channel.
    weight_dir: NN that outputs the context-dependent weight_dir vector.
    weight_amp: NN that outputs the context-dependent weight_amp vector.
    bias: NN that outputs the context-dependent bias vector.

  """

  def __init__(self, in_features, out_features, weight_dir, weight_amp, bias):
    assert(isinstance(weight_dir, torch.nn.Module))
    assert(isinstance(weight_amp, torch.nn.Module))
    assert(isinstance(bias, torch.nn.Module))
    self._input_features = in_features
    self._output_features = out_features
    super(CLinear, self).__init__()
    self._weight_dir = weight_dir
    self._weight_amp = weight_amp
    self._bias = bias
    self._in_features_cum = [0]
    self._out_features_cum = [0]
    in_c = 0
    out_c = 0
    for i in range(len(in_features)):
      in_c += in_features[i]
      out_c += out_features[i]
      self._in_features_cum.append(in_c)
      self._out_features_cum.append(out_c)
    mask_d = torch.zeros(self._out_features_cum[-1], self._in_features_cum[-1]).bool()
    mask_o = torch.zeros(self._out_features_cum[-1], self._in_features_cum[-1]).bool()
    for i in range(len(in_features)):
      mask_d[self._out_features_cum[i]:self._out_features_cum[i+1],self._in_features_cum[i]:self._in_features_cum[i+1]]=1
      mask_o[self._out_features_cum[i]:self._out_features_cum[i+1],0:self._in_features_cum[i]]=1
    self.register_buffer("mask_d", mask_d)
    self.register_buffer("mask_o", mask_o)

  def forward(self, x, c):
    weight_dir = self._weight_dir(c).view(-1,self._out_features_cum[-1],self._in_features_cum[-1])
    weight_amp = self._weight_amp(c)[:,:,None]
    bias = self._bias(c)[:,:,None]
    w = torch.zeros_like(weight_dir).where(~self.mask_d, weight_dir.exp()) + torch.zeros_like(weight_dir).where(~self.mask_o, weight_dir)
    squarednorm = (w ** 2).sum(dim=2, keepdim=True)
    w = w / squarednorm.sqrt()
    w = w * weight_amp.exp()
    _x = x[:,:,0,None]
    x_ = x[:,:,1,None]
    _x = (w @ _x) + bias
    logdiag = torch.empty_like(w).where(~self.mask_d, weight_dir - 0.5 * squarednorm.log() + weight_amp + x_.transpose(1,2).expand(-1,w.shape[1],-1))
    logdet = torch.zeros(_x.shape[0],0,1, device=_x.device)
    for i in range(len(self._input_features)):
      logdet = torch.cat((logdet, logdiag[:,self._out_features_cum[i]:self._out_features_cum[i+1],self._in_features_cum[i]:self._in_features_cum[i+1]].logsumexp(dim=2, keepdim=True)), dim=1)
    return torch.cat((_x, logdet), dim=2)
