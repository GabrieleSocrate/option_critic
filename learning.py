import theano, itertools
import numpy as np
from collections import OrderedDict
from theano import tensor

def get_learning_method(l_method, **kwargs):
    """
    Questa funzione serve a scegliere l’ottimizzatore (cioè la “regola” con cui si aggiornano i pesi della rete) in base a una stringa come "sgd" o "rmsprop".
    In pratica: tu dici “voglio RMSProp” e lei ti restituisce un oggetto RMSProp(...) pronto da usare.
    """
    if l_method == "adam":
      return Adam(**kwargs)
    elif l_method == "adadelta":
      return AdaDelta(**kwargs)
    elif l_method == "sgd":
      return SGD(**kwargs)
    elif l_method == "rmsprop":
      return RMSProp(**kwargs)

class SGD():
  def __init__(self, lr=0.01):
    self.lr = lr

  def apply(self, params, grads, grad_clip=0):
    updates = OrderedDict()
    for param, grad in zip(params, grads):
      if grad_clip > 0: # se grad_clip è attivo allora limita il gradiente in un intervallo e serve per evitare aggiornamenti instabili
          grad = tensor.clip(grad, -grad_clip, grad_clip)
      updates[param] = param - self.lr * grad # formula aggiornamento SGD (la batch size non c'è qui ma in realtà si usa un batch)
    return updates

class RMSProp():
  """
  RMSprop (Root Mean Square Propagation) è un algoritmo di ottimizzazione adattivo per il deep learning che velocizza l'addestramento dei modelli, 
  adattando il tasso di apprendimento (learning rate) per ogni parametro in base ai gradienti passati, usando una media mobile dei quadrati dei gradienti 
  per smorzare le oscillazioni e gestire meglio problemi non stazionari, risultando più efficiente di SGD e migliorando la stabilità. 
  """
  def __init__(self, rho=0.95, eps=1e-4, lr=0.001):
    self.rho = rho # quanto pesano i gradienti passati
    self.eps = eps 
    self.lr = lr
    print "rms", rho, eps, lr

  def apply(self, params, grads, grad_clip=0):
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        acc_grad = theano.shared(param.get_value() * 0) # variabile per memorizzare una media storica del gradiente
        acc_grad_new = self.rho * acc_grad + (1 - self.rho) * grad
        """
        La media esponenziale (EMA) è una media mobile che dà più peso ai valori recenti e sempre meno ai vecchi (decadimento esponenziale).
        Si aggiorna con: EMAₜ = α·xₜ + (1−α)·EMAₜ₋₁, dove α controlla quanto segue il dato nuovo.
        Se α è alto reagisce più velocemente, se α è basso è più stabile e “liscia”.
        È usata per smussare serie temporali e in ML/RL (es. Adam/RMSProp usano medie esponenziali).
        """

        acc_rms = theano.shared(param.get_value() * 0) # altra variabile per memorizzare la media del gradiente al quadrato
        acc_rms_new = self.rho * acc_rms + (1 - self.rho) * grad ** 2

        updates[acc_grad] = acc_grad_new
        updates[acc_rms] = acc_rms_new

        updates[param] = (param - self.lr * 
                          (grad / 
                           tensor.sqrt(acc_rms_new - acc_grad_new ** 2 + self.eps))) # questa è la formula di aggiornamento 

    return updates

class Adam():
  """
  Adam è un ottimizzatore che aggiorna i pesi usando:
  una media “storica” dei gradienti (1° momento)
  una media “storica” dei gradienti al quadrato (2° momento)
  In più fa una correzione di bias all’inizio (perché le medie partono da zero).
  """
  def __init__(self, lr=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-4):
    self.lr = lr
    self.b1 = beta1 # quanto memorizzi il passato nella media del gradiente 
    self.b2 = beta2 # quanto memorizzi il passato nella media del gradiente^2
    self.eps = epsilon

  def apply(self, params, grads):
    t = theano.shared(np.array(2., dtype='float32'))
    updates = OrderedDict()
    updates[t] = t+1
    for param, grad in zip(params, grads):
      last_1_moment = theano.shared(param.get_value() * 0) # variabile per la media storica dei gradienti
      last_2_moment = theano.shared(param.get_value() * 0) # variabile per la media storica dei gradienti^2
      new_last_1_moment = (1 - self.b1) * grad + self.b1 * last_1_moment
      new_last_2_moment = (1 - self.b2) * grad**2 + self.b2 * last_2_moment

      updates[last_1_moment] = new_last_1_moment
      updates[last_2_moment] = new_last_2_moment
      updates[param] = (param - (self.lr*(new_last_1_moment/(1-self.b1**t)) /
                  (tensor.sqrt(new_last_2_moment/(1-self.b2**t)) + self.eps)))#z.astype("float32")
      # Formula aggiornamento 
    return updates

class AdaDelta():
  """
  AdaDelta è un ottimizzatore che evita di scegliere un learning rate fisso in modo diretto: mantiene medie storiche di:
  gradienti^2
  update^2
  e scala gli update per renderli “consistenti”.
  """
  def __init__(self, rho=0.95, rho2=0.95):
    self.rho = rho
    self.rho2 = rho2

  def apply(self, params, grads):
    zipped_grads = [theano.shared(p.get_value() * 0) for p in params] # per memorizzare i gradienti
    running_up2 = [theano.shared(p.get_value() * 0) for p in params] # per memorizzare media storica degli update^2
    running_grads2 = [theano.shared(p.get_value() * 0) for p in params] # per memorizzare media storica dei gradienti^2

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, self.rho * rg2 + (1-self.rho) * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    #f_grad_shared = theano.function(input_params, cost, updates=zgup + rg2up,
    #                                name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, self.rho2 * ru2 + (1-self.rho2) * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    #f_update = theano.function([], [], updates=ru2up + param_up, name='adadelta_f_update')
    updates = ru2up + param_up

    return updates


