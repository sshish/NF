#######################################
#Basic interface of normalizing flows.#
#######################################

import torch

class Basic(torch.nn.Module):
  """Abstract module for normalizing flows.

  Methods:
    forward(x, c): Performs reversible transformation conditioned on
      context c. Input x and output y have shape BATCH*CHANNEL*INFO. INFO
      dimension has two entries: the first is value of the variable, the second
      gives information on log-determinant of the Jacobian of the
      transormation. Log-determinant is calculated by CHANNELwise summation of
      these second entries.
    inverse(y, c): Performs inverse transformation conditioned on context
      c. Input y and output x have shape BATCH*CHANNEL.

  """

  def __init__(self, *args):
    super(Basic, self).__init__()

  def forward(self, x, c):
    raise NotImplementedError

  def inverse(self, y, c):
    raise NotImplementedError

  def _get_name(self):
    return 'NF:' + super(Basic, self)._get_name()

class ToGenerator(torch.nn.Module):
  """Generative model created from a NF and a distribution over the latent
    variable.

  Args:
    net (NF): Module representing the normalizing flow.
    prior (torch.distributions.distribution.Distribution):Distribution over the
      latent variable y.

  Methods:
    sample(n=None, c=None): Samples n times conditioned on context c. Output
      shape is BATCH*n*CHANNEL. If c=None then BATCH dimension is omitted. If
      n=None then the sample dimension is omitted and only one sample is drawn.
    log_p(x, c=None): Calculates the model probability for a sample x
      conditioned on context c. Input x has shape BATCH*CHANNEL. Output shape
      is BATCH. c=None is treated as dummy variable with BATCH equal to x.
    crossentropy_loss(x, c=None): Returns a scalar loss suitable for training
      the model, based on Monte Carlo estimate of the crossentropy between data
      generating distribution and model distribution. Minimizing this is
      equivalent to minimizing the KL-divergence between aggregate posterior
      and prior distribution of the latent variable. Input x has shape
      BATCH*CHANNEL. c=None is treated as dummy variable with BATCH equal to x.
    entropy_loss(n, c=None): Returns a scalar loss based on Monte Carlo score
      function estimate of model distribution entropy gradient. Adding
      this term with a small negative factor to the loss function regularizes
      the network to have higher entropy in its output. n is number of samples
      for each element of c in BATCH dimension.
    entropy(n, c=None) Returns a Monte Carlo estimate of model distribution
      entropy. Output shape is BATCH. n is number of samples for each element
      of c in BATCH dimension. If c=None then BATCH dimension is omitted.

  """

  def __init__(self, net, prior):
    assert(isinstance(net, Basic))
    assert(isinstance(prior, torch.distributions.distribution.Distribution))
    super(ToGenerator, self).__init__()
    self._net = net
    self._prior = prior

  def _sample_unsqueezed(self, n, c):
    y = self._prior.sample(torch.Size([n]))
    y = self._net.inverse(y.repeat(c.shape[0],1), c.repeat_interleave(n, dim=0)).view(c.shape[0],n,-1)
    return y.detach()

  def sample(self, n=None, c=None):
    squeeze_n = False
    squeeze_c = False
    if n is None:
      squeeze_n = True
      n = 1
    if c is None:
      squeeze_c = True
      c = torch.empty(1)
    y = self._sample_unsqueezed(n, c)
    if squeeze_n:
      y = y.squeeze(dim=1)
    if squeeze_c:
      y = y.squeeze(dim=0)
    return y

  def log_p(self, x, c=None):
    x = torch.stack((x, torch.zeros_like(x)), dim=2)
    y = self._net(x, c)
    log_p1 = self._prior.log_prob(y[:,:,0])
    log_p2 = y[:,:,1].sum(dim=1)
    return log_p1 + log_p2

  def crossentropy_loss(self, x, c=None):
    return -1. * self.log_p(x, c).mean()

  def _entropy_unsqueezed(self, n, c):
    x = self._sample_unsqueezed(n, c)
    x = x.view(n * c.shape[0],-1)
    x = -1. * self.log_p(x, c.repeat_interleave(n, dim=0)).view(c.shape[0],n)#.mean(dim=1)
    return x

  def entropy(self, n, c=None):
    if c is None:
      squeeze_c = True
      c = torch.empty(1)
    x = self._entropy_unsqueezed(n, c).mean(dim=1)
    if squeeze_c:
      x = x.squeeze(dim=0)
    return x.detach()

  def entropy_loss(self, n, c=None):
    if c is None:
      c = torch.empty(1)
    b = self._entropy_unsqueezed(n, c).detach().mean(dim=1, keepdim=True)
    log_p = -1. * self._entropy_unsqueezed(n, c)
    a = log_p.detach()
    return -1. * ((a + b) * log_p).mean()

######################################################################
#Implementations of different layers that comply to the NF interface.#
######################################################################

class Stack(torch.nn.Sequential, Basic):
  """A NF that is created by stacking multiple NFs.

  Args: Individual NFs must be listed in the order from x-space (data) to y-
    space (latent variable).

  """

  def __init__(self, *args):
    for net in args:
      assert(isinstance(net, Basic))
    super(Stack, self).__init__(*args)

  def forward(self, x, c):
    for net in self._modules.values():
      x = net(x, c)
    return x

  def inverse(self, y, c):
    for net in reversed(self._modules.values()):
      y = net.inverse(y, c)
    return y

class Permutation(Basic):
  """Permutation of channels (context-free).

  Args:
    num_channels (int): The number of input (and output) channels.
    permutation: Specifies the permutation. Can be either 'random' (default) or
      'flip' or a 1D tensor of indices.

  """

  def __init__(self, num_channels, permutation="random"):
    assert(isinstance(num_channels, int))
    super(Permutation, self).__init__()
    self._num_channels = num_channels
    if permutation == "random":
      _permutation = torch.randperm(num_channels)
    elif permutation == "flip":
      _permutation = torch.arange(num_channels - 1, -1, -1)
    else:
      _permutation = permutation
    self.register_buffer("permutation", _permutation)
    _inverse_permutation = torch.empty_like(_permutation)
    for i in range(num_channels): _inverse_permutation[_permutation[i]] = i
    self.register_buffer("inverse_permutation", _inverse_permutation)

  def forward(self, x, _):
    return x[:,self.permutation]

  def inverse(self, y, _):
    return y[:,self.inverse_permutation]

class Rotation(Basic):
  """Context-free rotation of channels in the plane of channel1 and channel2.

  Args: channel1 (int), channel2 (int)

  """

  def __init__(self, channel1, channel2):
    assert(isinstance(channel1, int))
    assert(isinstance(channel2, int))
    super(Rotation, self).__init__()
    self._channel1 = channel1
    self._channel2 = channel2
    self._angle = torch.nn.Parameter(torch.Tensor([0]))

  def forward(self, x, _):
    result = x.clone()
    result[:,self._channel1,0] = x[:,self._channel1,0] * self._angle.cos() - x[:,self._channel2,0] * self._angle.sin()
    result[:,self._channel2,0] = x[:,self._channel1,0] * self._angle.sin() + x[:,self._channel2,0] * self._angle.cos()
    return result

  def inverse(self, y, _):
    result = y
    result[:,self._channel1] = y[:,self._channel1] * self._angle.cos() + y[:,self._channel2] * self._angle.sin()
    result[:,self._channel2] = - y[:,self._channel1] * self._angle.sin() + y[:,self._channel2] * self._angle.cos()
    return result

class CRotation(Basic):
  """Context-dependent rotation of channels in the plane of channel1 and
    channel2.

  Args: channel1 (int), channel2 (int), angle (torch.nn.Module)

  """

  def __init__(self, channel1, channel2, angle):
    assert(isinstance(channel1, int))
    assert(isinstance(channel2, int))
    assert(isinstance(angle, torch.nn.Module))
    super(CRotation, self).__init__()
    self._channel1 = channel1
    self._channel2 = channel2
    self._angle = angle

  def forward(self, x, c):
    result = x.clone()
    angle = self._angle(c).squeeze()
    result[:,self._channel1,0] = x[:,self._channel1,0] * angle.cos() - x[:,self._channel2,0] * angle.sin()
    result[:,self._channel2,0] = x[:,self._channel1,0] * angle.sin() + x[:,self._channel2,0] * angle.cos()
    return result

  def inverse(self, y, c):
    result = y
    angle = self._angle(c).squeeze(1)
    result[:,self._channel1] = y[:,self._channel1] * angle.cos() + y[:,self._channel2] * angle.sin()
    result[:,self._channel2] = - y[:,self._channel1] * angle.sin() + y[:,self._channel2] * angle.cos()
    return result

class Clamp(Basic):
  """Applicable when inputs are bounded between low and high. Useful for
    ensuring that inverse transformation is also bounded.

  Args: low, high: vectors describing minimal and maximal x-values.

  """

  def __init__(self, low, high):
    super(Clamp, self).__init__()
    self.register_buffer("low", low)
    self.register_buffer("high", high)

  def forward(self, x, _):
    return x

  def inverse(self, y, _):
    return y.where(y > self.low, self.low).where(y < self.high, self.high)


class Tanh(Basic):
  """Tanh layer with NF interface (context-free).

  """

  def __init__(self):
    super(Tanh, self).__init__()

  def forward(self, x, _):
    _x = x[:,:,0]
    delta = - 2. * (_x - torch.empty_like(_x).fill_(2).log() + torch.nn.functional.softplus(- 2. * _x))
    return torch.stack((_x.tanh(), x[:,:,1] + delta), dim=2)

  def inverse(self, y, _):
    return 0.5 * ((1. + y) / (1. - y)).log()

class Atanh(Basic):
  """Atanh layer with NF interface (context-free). Useful for ensuring that
    inverse transformation is bounded.

  Args: low, high: vectors describing minimal and maximal x-values.

  """

  def __init__(self, low, high):
    super(Tan, self).__init__()
    inflection = 0.5 * (high + low)
    steepness = 2 / (high - low)
    self.register_buffer("inflection", inflection)
    self.register_buffer("steepness", steepness)

  def forward(self, x, _):
    _x = x[:,:,0]
    _x = self.steepness * (_x - self.inflection)
    return torch.stack((0.5 * ((1. + _x) / (1. - _x)).log(), x[:,:,1] - (1. - _x ** 2).log()), dim=2)

  def inverse(self, y, _):
    return y.tanh() / self.steepness + self.inflection

class CouplingLayer(Basic):
  """Context-free coupling layer from the RealNVP paper.

  """

  def __init__(self, s, t, mask):
    assert(isinstance(s, torch.nn.Module))
    assert(isinstance(t, torch.nn.Module))
    assert(isinstance(mask, torch.BoolTensor))
    super(CouplingLayer, self).__init__()
    self._s = s
    self._t = t
    self.register_buffer("mask", mask)

  def forward(self, x, _):
    _x = x[:,:,0]
    _x_ = torch.where(self.mask, _x, torch.zeros_like(_x))
    scale = torch.where(~self.mask, self._s(_x_), torch.zeros_like(_x_))
    trans = torch.where(~self.mask, self._t(_x_), torch.zeros_like(_x_))
    _x = _x.where(self.mask, _x * scale.exp() + trans)
    return torch.stack((_x, x[:,:,1] + scale), dim=2)

  def inverse(self, y, _):
    y_ = torch.where(self.mask, y, torch.zeros_like(y))
    scale = torch.where(~self.mask, self._s(y_), torch.zeros_like(y_))
    trans = torch.where(~self.mask, self._t(y_), torch.zeros_like(y_))
    y = y.where(self.mask, (y - trans) * (-1. * scale).exp())
    return y

class CCouplingLayer(Basic):
  """Context-dependent coupling layer.

  """

  def __init__(self, s, t, mask):
    assert(isinstance(s, torch.nn.Module))
    assert(isinstance(t, torch.nn.Module))
    assert(isinstance(mask, torch.BoolTensor))
    super(CCouplingLayer, self).__init__()
    self._s = s
    self._t = t
    self.register_buffer("mask", mask)

  def forward(self, x, c):
    _x = x[:,:,0]
    _x_ = torch.where(self.mask, _x, torch.zeros_like(_x))
    scale = torch.where(~self.mask, self._s(_x_, c), torch.zeros_like(_x_))
    trans = torch.where(~self.mask, self._t(_x_, c), torch.zeros_like(_x_))
    _x = _x.where(self.mask, _x * scale.exp() + trans)
    return torch.stack((_x, x[:,:,1] + scale), dim=2)

  def inverse(self, y, c):
    y_ = torch.where(self.mask, y, torch.zeros_like(y))
    scale = torch.where(~self.mask, self._s(y_, c), torch.zeros_like(y_))
    trans = torch.where(~self.mask, self._t(y_, c), torch.zeros_like(y_))
    y = y.where(self.mask, (y - trans) * (-1. * scale).exp())
    return y
