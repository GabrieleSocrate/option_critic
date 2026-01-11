import theano, sys, copy
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from nnet import Model
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

class MLP3D():
  """
  In Option-Critic hai più opzioni o∈{0,…,K−1}.
  Ogni opzione ha una intra-option policy π(a∣s,o), cioè:
  “dato lo stato, quale azione scegliere se sono nell’opzione o?”
  MLP3D implementa proprio questo: invece di avere una sola matrice di pesi per tutte le opzioni, ha un set di pesi per ogni opzione. 
  options_W: pesi diversi per ogni opzione
  options_b: bias diversi per ogni opzione
  apply(inputs, option): dato lo stato e l’opzione, calcola una distribuzione sulle azioni (softmax)
  Il nome “3D” viene dal fatto che options_W ha 3 dimensioni:
  (num_options, hidden_size, num_actions)
  """
  def __init__(self, num_options, model_network, temp=1):
    self.temp = temp
    self.options_W = theano.shared(np.random.uniform(
        size=(num_options, model_network[-2]["out_size"], model_network[-1]["out_size"]), high=1, low=-1))
    """
    Qui crea i pesi per tutte le opzioni:
    theano.shared(...): crea una variabile “trainabile” per Theano.
    np.random.uniform(..., low=-1, high=1): inizializza i pesi random tra -1 e 1.
    size=(num_options, ..., ...):
    dimensione 0: num_options (un blocco per ogni opzione)
    dimensione 1: model_network[-2]["out_size"] = la dimensione dell’input che arriva all’MLP (tipicamente 512)
    dimensione 2: model_network[-1]["out_size"] = numero di output (tipicamente numero azioni)
    """
    self.options_b = theano.shared(np.zeros((num_options, model_network[-1]["out_size"])))
    # bias per ogni opzione
    self.params = [self.options_W, self.options_b]

  def apply(self, inputs, option):
    """
    Dato:
    inputs = feature dello stato (batch)
    option = quale opzione è attiva
    calcola:
    la distribuzione sulle azioni 
    π(a∣s,o) con softmax.
    """
    W = self.options_W[option] # seleziona i pesi dell'opzione specifica 
    b = self.options_b[option] # seleziona bias dell'opzione specifica

    dots = T.sum(inputs.dimshuffle(0,1,'x')*W, axis=1) # questo ti da la moltiplicazione tra inputs e W 
    return T.nnet.softmax((dots + b)/self.temp)
  """
  Aggiunge il bias b.
  Divide per temp (temperatura):
  temp bassa → softmax più “decisa” (più vicina a one-hot)
  temp alta → softmax più “piatta” (più esplorazione)
  Applica softmax → ottieni probabilità sulle azioni.
  Quindi questa è la policy intra-opzione: una distribuzione sulle azioni.
  """

  def save_params(self):
    return [i.get_value() for i in self.params]

  def load_params(self, values):
    print "LOADING NNET..",
    for p, value in zip(self.params, values):
      p.set_value(value)
    print "LOADED"

class OptionCritic_Network():
  def __init__(self, model_network=None, gamma=0.99, learning_method="rmsprop", actor_lr=0.00025,
    batch_size=32, input_size=None, learning_params=None, dnn_type=True, clip_delta=0,
    scale=255., freeze_interval=100, grad_clip=0, termination_reg=0, num_options=8,
    double_q=False, temp=1, entropy_reg=0, BASELINE=False, **kwargs):
    x = T.ftensor4() # batch di stati
    next_x = T.ftensor4() # batch di next state
    a = T.ivector() # vettore di azioni
    o = T.ivector() # vettore di opzioni
    r = T.fvector() # vettore di reward 
    terminal = T.ivector() # vettore done (0/1)
    self.freeze_interval = freeze_interval # ogni quanto aggiorni la target network

    self.theano_rng = MRG_RandomStreams(1000) # serve per campionare termination e azioni

    self.x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32')) # contengono batch di immagini
    self.next_x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32')) # contengono batch di immagini
    self.a_shared = theano.shared(np.zeros((batch_size), dtype='int32')) # contengono batch di azioni
    self.o_shared = theano.shared(np.zeros((batch_size), dtype='int32')) # contengono batch di opzioni
    self.terminal_shared = theano.shared(np.zeros((batch_size), dtype='int32')) # contengono batch di done
    self.r_shared = theano.shared(np.zeros((batch_size), dtype='float32')) # contengono batch di reward

    state_network = model_network[:-1] # Prende tutto tranne l’ultimo layer: quindi ottieni la parte che trasforma immagini → feature (es. 512).
    termination_network = copy.deepcopy([model_network[-1]]) # Crea una copia dell’ultimo layer e lo userà come “testa” per termination.
    termination_network[0]["activation"] = "sigmoid" # Termination deve essere probabilità in (0,1) → sigmoid.
    print "NUM OPTIONS --->", num_options
    termination_network[0]["out_size"] = num_options # l'output size deve essere pari al numero di opzioni dato che devo avere una probabilità per ogni opzione 
    # ora facciamo una roba simile a quella fatta per la termination network
    option_network = copy.deepcopy([model_network[-1]])
    option_network[0]["activation"] = "softmax" # il valore Q è un numero reale non tra 0,1 quindi scelgo softmax come activation function
    Q_network = copy.deepcopy([model_network[-1]])
    Q_network[0]["out_size"] = num_options # anche qui l'ouput size dive essere pari al numero di opzioni dato che devo avere un valore Q per ogni opzione

    self.state_model = Model(state_network, input_size=input_size, dnn_type=dnn_type)
    self.state_model_prime = Model(state_network, input_size=input_size, dnn_type=dnn_type) # questa è una rete target che viene aggiornata ogni freeze_interval
    output_size = [None,model_network[-2]["out_size"]]
    self.Q_model = Model(Q_network, input_size=output_size, dnn_type=dnn_type)
    self.Q_model_prime = Model(Q_network, input_size=output_size, dnn_type=dnn_type) # questa è una rete target che viene aggiornata ogni freeze_interval
    self.termination_model = Model(termination_network, input_size=output_size, dnn_type=dnn_type)
    self.options_model = MLP3D(num_options, model_network, temp=temp)

    s = self.state_model.apply(x/scale)
    next_s = self.state_model.apply(next_x/scale)
    next_s_prime = self.state_model_prime.apply(next_x/scale)
    """
    next_s e next_s_prime sono la rappresentazione numerica (feature vector) del prossimo stato, ottenuta passando next_x (stack di 4 frame 84×84) nella CNN+MLP.
    Entrambi hanno la stessa dimensione (tipicamente 512 per ogni elemento del batch), perché l’architettura è la stessa.
    La differenza è nei pesi: state_model è la rete online aggiornata continuamente, state_model_prime è la rete target più stabile, aggiornata solo ogni freeze_interval.
    """

    termination_probs = self.termination_model.apply(theano.gradient.disconnected_grad(s)) # termination_probs ha shape (batch, num_options)
    option_term_prob = termination_probs[T.arange(o.shape[0]), o] # prendi per ogni elemento del batch la prob della sua opzione
    # ora facciamo lo stesso ma nel next state
    next_termination_probs = self.termination_model.apply(theano.gradient.disconnected_grad(next_s)) 
    next_option_term_prob = next_termination_probs[T.arange(o.shape[0]), o]
    termination_sample = T.gt(option_term_prob, self.theano_rng.uniform(size=o.shape))
    """
    uniform(...) genera numeri random tra 0 e 1 (uno per elemento batch)
    T.gt(prob, u) significa “prob > u?”
    quindi termination_sample è un booleano: True se termina, False se continua
    """
    Q = self.Q_model.apply(s) # Q shape (batch, num_options). Contiene Q-value per ogni opzione
    next_Q = self.Q_model.apply(next_s) # Q nel next state usando rete online
    next_Q_prime = theano.gradient.disconnected_grad(self.Q_model_prime.apply(next_s_prime)) # Q nel next state usando rete target

    disc_option_term_prob = theano.gradient.disconnected_grad(next_option_term_prob)

    action_probs = self.options_model.apply(s, o)
    sampled_actions = T.argmax(self.theano_rng.multinomial(pvals=action_probs, n=1), axis=1).astype("int32")
    """
    IMPORTANTE!!!!!!!
    In Option-Critic, nel next state puoi:
    continuare la stessa opzione con probabilità 1−β(s′,o)
    terminare e scegliere la migliore opzione con probabilità β(s′,o)
    Quindi il target del critic è una media pesata di due termini.
    """
    if double_q:
      """
      Double Q (Double DQN) serve a evitare che il max sui Q-values li sovrastimi.
      Fa così: usa la rete online per scegliere l’azione/opzione migliore (argmax), ma usa la rete target per valutarne il valore Q.
      """
      print "TRAINING DOUBLE_Q"
      y = r + (1-terminal)*gamma*(
        (1-disc_option_term_prob)*next_Q_prime[T.arange(o.shape[0]), o] +
        disc_option_term_prob*next_Q_prime[T.arange(next_Q.shape[0]), T.argmax(next_Q, axis=1)])
      """
      r = reward immediato
      (1-terminal) = se terminal=1 (episodio finito), allora non aggiungere futuro
      gamma = discount
      Dentro la parentesi:
      continua stessa opzione:
      (1 - β) * Q'(s', o)
      next_Q_prime[range, o] prende la Q target della stessa opzione nel next state
      termina e scegli best opzione:
      β * Q'(s', o*)
      ma con Double-Q:
      scegli o* = argmax(next_Q) usando la rete online
      valuti con next_Q_prime (target)
      Questo riduce l’overestimation.
      """
    else:
      """
      Se non usi Double Q, fai la cosa “classica”:
      scegli l’azione/opzione migliore e la valuti usando la stessa rete target con un max:
      y=r+γ max(respect to a) Q_target(s',a)
      """
      y = r + (1-terminal)*gamma*(
        (1-disc_option_term_prob)*next_Q_prime[T.arange(o.shape[0]), o] +
        disc_option_term_prob*T.max(next_Q_prime, axis=1))

    y = theano.gradient.disconnected_grad(y)

    option_Q = Q[T.arange(o.shape[0]), o] # Prende Q(s,o) per l’opzione corrente, per ogni elemento del batch.
    td_errors = y - option_Q # TD error per ogni elemento del batch.
    """
    option_Q = la tua stima corrente Q(s,o;θ) (output della rete sullo stato corrente e l’opzione corrente).
    y = la stima/target di Bellman
    Quindi: la “stima di Bellman” è y, mentre option_Q è la previsione attuale che stai cercando di correggere.
    """

    if clip_delta > 0: # questa è la huber loss 
      """
      Huber fa questo:
      se td error è piccolo → usa una loss quadratica (come MSE)
      se td error è grande → passa a una loss lineare (cresce più lentamente)
      """
      quadratic_part = T.minimum(abs(td_errors), clip_delta)
      linear_part = abs(td_errors) - quadratic_part
      td_cost = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
    else:
      td_cost = 0.5 * td_errors ** 2 # questo invece è MSE loss 0.5 perchè coaì la derivata è td_error e non 2 * td_error

    #critic updates
    critic_cost = T.sum(td_cost) # sommo la loss (td_cost) su tutto il batch
    critic_params = self.Q_model.params + self.state_model.params
    learning_algo = self.Q_model.get_learning_method(learning_method, **learning_params) # scelgo l'ottimizzatore 
    grads = T.grad(critic_cost, critic_params) # calcola i gradienti della los rispetto a quei parametri 
    critic_updates = learning_algo.apply(critic_params, grads, grad_clip=grad_clip)

    #actor updates
    actor_params = self.termination_model.params + self.options_model.params
    learning_algo = self.termination_model.get_learning_method("sgd", lr=actor_lr) # per l'actor uso sgd
    disc_Q = theano.gradient.disconnected_grad(option_Q)
    disc_V = theano.gradient.disconnected_grad(T.max(Q, axis=1))
    # disc_Q = Q(s,o) per l’opzione corrente 
    # disc_V = maxo′ Q(s,o′) (il “miglior valore” tra tutte le opzioni nello stato)
    term_grad = T.sum(option_term_prob*(disc_Q-disc_V+termination_reg))
    """
    Serve a insegnare a β quando:
    conviene continuare con l’opzione corrente
    conviene terminare e passare a un’altra opzione
    Come lo fa:
    option_term_prob = β(s,o): più è alta, più “spingi” verso la terminazione.
    disc_Q - disc_V confronta:
    Q(s,o) = quanto è buona l’opzione corrente
    V(s)=maxo′ Q(s,o′) = quanto sarebbe buona la migliore opzione
    Se l’opzione corrente è peggiore della migliore, allora disc Q − disc V
    è negativo → il gradiente tende a far aumentare β → termini più spesso.
    termination_reg è un bias che controlla quanto “terminare” è incentivato o penalizzato (quindi opzioni più corte o più lunghe).
    In una frase: term_grad è il pezzo che allena β(s,o) a cambiare opzione quando l’opzione corrente non è conveniente.
    """
    entropy = -T.sum(action_probs*T.log(action_probs)) # calcola l’entropia della policy sulle azioni π(a|s,o).
    if not BASELINE:
      policy_grad = -T.sum(T.log(action_probs[T.arange(a.shape[0]), a]) * y) - entropy_reg*entropy
    else:
      policy_grad = -T.sum(T.log(action_probs[T.arange(a.shape[0]), a]) * (y-disc_Q)) - entropy_reg*entropy
    grads = T.grad(term_grad+policy_grad, actor_params)
    actor_updates = learning_algo.apply(actor_params, grads, grad_clip=grad_clip)

    if self.freeze_interval > 1:
      target_updates = OrderedDict()
      for t, b in zip(self.Q_model_prime.params+self.state_model_prime.params,
                        self.Q_model.params+self.state_model.params):
        target_updates[t] = b
      self._update_target_params = theano.function([], [], updates=target_updates)
      self.update_target_params()
      print "freeze interval:", self.freeze_interval
    else:
      print "freeze interval: None"

    critic_givens = {x:self.x_shared, o:self.o_shared, r:self.r_shared,
    terminal:self.terminal_shared, next_x:self.next_x_shared}

    actor_givens = {a:self.a_shared, r:self.r_shared,
    terminal:self.terminal_shared, o:self.o_shared, next_x:self.next_x_shared}

    print "compiling...",
    self.train_critic = theano.function([], [critic_cost], updates=critic_updates, givens=critic_givens)
    self.train_actor = theano.function([s], [], updates=actor_updates, givens=actor_givens)
    self.pred_score = theano.function([], T.max(Q, axis=1), givens={x:self.x_shared})
    self.sample_termination = theano.function([s], [termination_sample,T.argmax(Q, axis=1)], givens={o:self.o_shared})
    self.sample_options = theano.function([s], T.argmax(Q, axis=1))
    self.sample_actions = theano.function([s], sampled_actions, givens={o:self.o_shared})
    self.get_action_dist = theano.function([s, o], action_probs)
    self.get_s = theano.function([], s, givens={x:self.x_shared})
    print "complete"

  def update_target_params(self):
    if self.freeze_interval > 1:
      self._update_target_params()
    return

  def predict_move(self, s):
    return self.sample_options(s)

  def predict_termination(self, s, a):
    self.a_shared.set_value(a)
    return tuple(self.sample_termination(s))

  def get_q_vals(self, x):
    self.x_shared.set_value(x)
    return self.pred_score()[:,np.newaxis]

  def get_state(self, x):
    self.x_shared.set_value(x)
    return self.get_s()

  def get_action(self, s, o):
    self.o_shared.set_value(o)
    return self.sample_actions(s)

  def train_conv_net(self, train_set_x, next_x, options, r, terminal, actions=None, model=""):
    self.next_x_shared.set_value(next_x)
    self.o_shared.set_value(options)
    self.r_shared.set_value(r)
    self.terminal_shared.set_value(terminal)
    if model == "critic":
        self.x_shared.set_value(train_set_x)
        return self.train_critic()
    elif model == "actor":
      self.a_shared.set_value(actions)
      return self.train_actor(train_set_x)
    else:
      print "WRONG MODEL NAME"
      raise NotImplementedError

  def save_params(self):
    return [self.state_model.save_params(), self.Q_model.save_params(), self.termination_model.save_params(),
    self.options_model.save_params()]

  def load_params(self, values):
    self.state_model.load_params(values[0])
    self.Q_model.load_params(values[1])
    self.termination_model.load_params(values[2])
    self.options_model.load_params(values[3])
