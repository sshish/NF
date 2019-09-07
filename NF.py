#######################################
#Basic interface of normalizing flows.#
#######################################

import torch

class Basic(torch.nn.Module):
  """Abstract module for normalizing flows.

  Methods:
    forward(x): Performs inference of the latent variable conditioned on the
      input x. Returns the value of the latent variable y, as well as the
      logarithm of the absolute value of the determinant of the Jacobian. The
      dimensions of the input and return values are BATCHxCHANNELxINFO. The INFO
      dimension has two entries: the first is for return value, the second
      provides information about the log-determinant. The log-determinant is
      calculated by summation of these second entries in the INFO dimension over
      the CHANNEL dimension. The benefit of this representation (rather than
      just storing the entire log-determinant in a single location) is that it
      allows to split the input before passing it to different consecutive flows
      and concatenating back again.
    inverse(y): Acts in the opposite direction to the forward method and is for
      sampling x values.

  """

  def __init__(self, *args):
    super(Basic, self).__init__()

  def inverse(self, y):
    raise NotImplementedError

class ToGenerator(torch.nn.Module):
  """Generative model created from a NF and a distribution over the latent
    variable.

  Args:
    net (NF): The module representing the normalizing flow.
    prior (torch.distributions.distribution.Distribution): The distribution over
      the latent variable y.

  Methods:
    sample(n): Creates n samples.
    log_p(x): Calculates the model probability for a sample x.
    crossentropy_loss(x): Returns a loss tensor suitable for training the model,
      based on the Monte Carlo estimate of the crossentropy between data
      generating distribution and model distribution. Minimizing this is
      equivalent to minimizing the KL-divergence between aggregate posterior and
      prior distribution of the latent variable.
    entropy_loss(n): Returns a loss tensor that corresponds to a Monte Carlo
      estimate of the model distribution entropy plus a constant term. Adding
      this term with a small negative factor to the loss function regularizes
      the network to have higher entropy in its output. n is the number of
      samples for the Monte Carlo estimate.
    entropy(n) Returns Monte Carlo estimate of model distribution entropy.
      n is number of samples.

  """

  def __init__(self, net, prior):
    assert(isinstance(net, Basic))
    assert(isinstance(prior, torch.distributions.distribution.Distribution))
    super(ToGenerator, self).__init__()
    self._net = net
    self._prior = prior

  def sample(self, n):
    y = self._prior.sample(torch.Size([n]))
    y = self._net.inverse(y)
    return y.detach()

  def log_p(self, x):
    x = torch.stack((x, torch.zeros_like(x)), dim=2)
    y = self._net(x)
    log_p1 = self._prior.log_prob(y[:,:,0])
    log_p2 = y[:,:,1].sum(dim=1)
    return log_p1 + log_p2

  def crossentropy_loss(self, x):
    return -1. * self.log_p(x).mean()

  def entropy(self, n):
    x = self.sample(n)
    return self.crossentropy_loss(x).detach()

  def entropy_loss(self, n):
    x = self.sample(n)
    log_p = self.log_p(x)
    A = log_p.detach()
    return -1. * ((A + self.entropy(n)) * log_p).mean()

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

  def inverse(self, y):
    for net in reversed(self._modules.values()):
      y = net.inverse(y)
    return y

class Permutation(Basic):
  """Permutation of channels.

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

  def forward(self, x):
    return x[:,self.permutation]

  def inverse(self, y):
    return y[:,self.inverse_permutation]

class Rotation(Basic):
  """Rotation of channels in the plane of channel1 and channel2.

  Args: channel1 (int), channel2 (int)

  """

  def __init__(self, channel1, channel2):
    assert(isinstance(channel1, int))
    assert(isinstance(channel2, int))
    super(Rotation, self).__init__()
    self._channel1 = channel1
    self._channel2 = channel2
    self._angle = torch.nn.Parameter(torch.Tensor([0]))

  def forward(self, x):
    result = x.clone()
    result[:,self._channel1,0] = x[:,self._channel1,0] * self._angle.cos() - x[:,self._channel2,0] * self._angle.sin()
    result[:,self._channel2,0] = x[:,self._channel1,0] * self._angle.sin() + x[:,self._channel2,0] * self._angle.cos()
    return result

  def inverse(self, y):
    result = y
    result[:,self._channel1] = y[:,self._channel1] * self._angle.cos() + y[:,self._channel2] * self._angle.sin()
    result[:,self._channel2] = - y[:,self._channel1] * self._angle.sin() + y[:,self._channel2] * self._angle.cos()
    return result

class Tanh(Basic):
  """Tanh layer with NF interface.

  """

  def __init__(self):
    super(Tanh, self).__init__()

  def forward(self, x):
    _x = x[:,:,0]
    delta = - 2. * (_x - torch.empty_like(_x).fill_(2).log() + torch.nn.functional.softplus(- 2. * _x))
    return torch.stack((_x.tanh(), x[:,:,1] + delta), dim=2)

  def inverse(self, y):
    return 0.5 * ((1. + y) / (1. - y)).log()

class CouplingLayer(Basic):
  """Coupling layer from the RealNVP paper.

  """

  def __init__(self, s, t, mask):
    assert(isinstance(s, torch.nn.Module))
    assert(isinstance(t, torch.nn.Module))
    assert(isinstance(mask, torch.BoolTensor))
    super(CouplingLayer, self).__init__()
    self._s = s
    self._t = t
    self.register_buffer("mask", mask)

  def forward(self, x):
    _x = x[:,:,0]
    _x_ = torch.where(self.mask, _x, torch.zeros_like(_x))
    scale = torch.where(~self.mask, self._s(_x_), torch.zeros_like(_x_))
    trans = torch.where(~self.mask, self._t(_x_), torch.zeros_like(_x_))
    _x = _x.where(self.mask, _x * scale.exp() + trans)
    return torch.stack((_x, x[:,:,1] + scale), dim=2)

  def inverse(self, y):
    y_ = torch.where(self.mask, y, torch.zeros_like(y))
    scale = torch.where(~self.mask, self._s(y_), torch.zeros_like(y_))
    trans = torch.where(~self.mask, self._t(y_), torch.zeros_like(y_))
    y = y.where(self.mask, (y - trans) * (-1. * scale).exp())
    return y
