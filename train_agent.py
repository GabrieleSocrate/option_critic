import sys, pdb, time, random, os, datetime, csv, theano, copy, pickle
import cv2
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import pickle as pkl
import theano.tensor as T
import scipy, scipy.misc
from neural_net import OptionCritic_Network
from exp_replay import DataSet
from plot_learning import plot

sys.setrecursionlimit(50000)


def load_params(model_path):
  mydir = "/".join(model_path.split("/")[:-1])
  """
  Prende il percorso del modello e ne estrae solo la cartella.
  Esempio: models/run1/last_model.pkl → models/run1
  """
  model_params = pkl.load(open(os.path.join(mydir, 'model_params.pkl'), 'rb'))
  return model_params

def create_dir(p):
  try:
    os.makedirs(p)
  except OSError, e:
    if e.errno != 17:
      raise # This was not a "directory exist" error..
  """
  os.makedirs(p) crea la directory p (anche se servono più livelli).
  except OSError, e: cattura errori di filesystem.
  if e.errno != 17:
  17 (su molti sistemi) significa “File exists / directory already exists”.
  Quindi:
  se l’errore è “esiste già” → ok, ignoralo
  se è un altro errore → raise (fallisci)
  """

def filecreation(model_params, folder_name=None):
  tempdir = os.path.join(os.getcwd(), "models")
  create_dir(tempdir)
  folder_name = folder_name if folder_name is not None else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  mydir = os.path.join(tempdir, folder_name)
  create_dir(mydir)
  pkl.dump(model_params, open(os.path.join(mydir, 'model_params.pkl'), "wb"))
  return mydir

class Trainer(object):
  def create_results_file(self):
    """
    scopo generale: Creare file di output dove salvare i risultati del training
    """
    self.prog_file = os.path.join(self.mydir, 'training_progress.csv')
    data_file = open(self.prog_file, 'wb')
    data_file.write('epoch,mean_score,mean_q_val\n')
    data_file.close()

    self.term_prob_file = os.path.join(self.mydir, 'term_prob.csv') # in questo secondo file mettiamo le termination probability
    data_file = open(self.term_prob_file, 'wb')
    data_file.write('epoch,termination_prob\n')
    data_file.close()

  def update_results(self, epoch, ave_reward, ave_q):
    """
    scopo generale: aggiornare il file CSV, creato con la funzione prima, durante l'addestramento, cioè salvare i progressi
    """
    # if it isn't, then we are testing and watching a game.
    # no need to update a file.
    if self.params.nn_file is None:
      """
      nn_file è un file di una rete pre-addestrata (--nn-file). 
      Quindi se è None, significa che stai facendo training da zero e quindi aggiorni il file
      Se invece è None, allora è già addestrata e quindi non fai nulla
      """
      fd = open(self.prog_file,'a') # dato che stiamo aggiornando usiamo la modalità append (a)
      fd.write('%d,%f,%f\n' % (epoch, ave_reward, ave_q))
      fd.close()
      plot(self.mydir)

  def update_term_probs(self, epoch, term_probs):
    """
    Scopo generale: aggiornare il secondo file CSV con la termination probability delle opzioni durante il training 
    In Option-Critic, la termination function β(s,o) è centrale: vuoi monitorare se:
    termina sempre subito (opzioni inutili)
    non termina mai (opzioni troppo lunghe)
    """
    if self.params.nn_file is None: # anche qui fa l'aggiornamento solo se stiamo trainando da 0
      fd = open(self.term_prob_file,'a')
      term_probs = term_probs if type(term_probs) is list else [term_probs] # assicura che term_probs sia una lista.
      for term_prob in term_probs:
        fd.write('%d,%f\n' % (epoch, term_prob))
      fd.close()

  def test_dnn(self): # non importante 
    #chooses which convnet to use based on cudnn availability
    self.params.USE_DNN_TYPE = False
    if theano.config.device.startswith("gpu"):
      self.params.USE_DNN_TYPE=theano.sandbox.cuda.dnn.dnn_available()
    if self.params.USE_DNN_TYPE:
      print "USING CUDNN"
    else:
      print "WARNING: NOT USING CUDNN. TRAINING WILL BE SLOWER."
    #self.params.USE_DNN_TYPE=False

  def __init__(self, model_params, ale_env, folder_name): # ale_env è l'ambiente Atari
    """
    Scopo generale:
    Il costruttore __init__ serve a preparare tutto prima del training:
    creare cartelle e file di log
    caricare la ROM Atari nell’emulatore
    impostare parametri dell’ambiente
    inizializzare variabili utili (random, contatori, dimensioni schermo, ecc.)
    """
    self.init_time = time.time()
    # nn_file only present when watching test
    if model_params.nn_file is None:
      self.mydir = filecreation(model_params, folder_name)
      self.create_results_file()

    self.params = model_params

    #ale_env.setInt('frame_skip', self.params.frame_skip)
    ale_env.setFloat('repeat_action_probability', 0.)
    ale_env.setBool('color_averaging', self.params.mean_frame)
    ale_env.loadROM(self.params.rom_path)

    self.print_option_stats = model_params.testing
    self.term_ratio = 0

    self.test_dnn()

    self.rng = np.random.RandomState(1234)
    self.noop_action = 0 # lazione no-op signfica non fare niente ed è l'azione per iniziare l'episodio

    self.frame_count = 0. # conta gli step (frame) fatti
    self.best_reward = -100. # tiene traccia del miglior reward visto fin'ora e lo inzializza a -100
    self.max_frames_per_game = 18000 # limite massimo di frame per episodio (per non far durare episodi all'infinito)

    self.ale = ale_env
    self.legal_actions = self.ale.getMinimalActionSet()
    print "NUM ACTIONS --->", len(self.legal_actions)
    self.screen_dims = self.ale.getScreenDims()
    self.last_dist = 0
    self.mean_entropy = 0

    self.action_counter = [{j:0 for j in self.legal_actions} for i in range(self.params.num_options)]
    # crea una lista lunga num_options, in cui ogni elemento è un dizionario che conta quante volte ogni azione è stata scelta quando era
    # attiva l'opzione

  def cap_reward(self, reward, testing=False):
    """
    Scopo generale: reward clipping ovvero trasforma reward reali in -1, 0, 1, per stabilità (molto comune per Atari)
    """
    if self.params.do_cap_reward and not testing:
      if reward > 0:
        score = 1
      elif reward < 0:
        score = -1
      else:
        score = 0
      return score
    else:
      return reward

  def _init_ep(self):
    """
    Scopo generale: 
    All’inizio di un episodio, Atari spesso fa:
    un numero casuale di “no-op” (azioni nulla) per iniziare da stati diversi,
    e costruisce lo stato iniziale come stack di frame
    """
    num_actions = np.random.randint(4, self.params.max_start_nullops)
    x = []
    self.last_screen = np.zeros((210, 160), dtype='uint8')
    for i in range(num_actions):
      self.ale.act(self.noop_action)
      if i >= num_actions-self.params.phi_length:
        x.append(self.get_observation())
    return x

  def act(self, action, testing=False):
    """
    Scopo generale:
    esegue un'azione nell'ambiente e ritorna reward e osservazione successiva
    """
    reward = 0 # accumulatore del reward totale
    for i in range(self.params.frame_skip):
      reward += self.ale.act(self.legal_actions[action])
    x = self.get_observation()
    return self.cap_reward(reward, testing), self.cap_reward(reward, True), x

  def get_observation(self): # Funzione specifica per gioco Atari
    screen = self.ale.getScreenGrayscale().reshape(self.screen_dims[1], self.screen_dims[0])

    if self.params.resize_method == "crop":
      resized = scipy.misc.imresize(screen, size=(110,84))[self.params.offset:self.params.offset+84, :]
    elif self.params.resize_method == "scale":
      resized = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_LINEAR)
    else:
      print "wrong resize_method, only have crop and scale"
      raise NotImplementedError
    return resized

  def save_model(self, total_reward, skip_best=False):
    """
    Scopo generale:
    Durante il training vuoi salvare:
    l’ultimo modello (“last”) per riprendere il training
    il miglior modello (“best”) per tenere quello che performa meglio
    È una pratica standard: se un training peggiora dopo, almeno hai salvato il migliore.
    """
    if total_reward >= self.best_reward and not skip_best:
      self.best_reward = total_reward
      pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'best_model.pkl'), "w"), protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'last_model.pkl'), "w"), protocol=pkl.HIGHEST_PROTOCOL)
    print "Saved model"

  def run_training_episode(self):
    raise NotImplementedError

  def get_learning_params(self):
    """
    Scopo generale: Costruisce un dizionario con i parametri necessari all’ottimizzatore (learning rule).
    """
    d = {}
    if self.params.update_rule == "rmsprop":
      """
      se l'update rule è rmsprop allora ci sono 3 parameteri che sono il learning rate, epsilon e decay rate
      """
      d["lr"] = self.params.learning_rate
      d["eps"] = self.params.rms_epsilon
      d["rho"] = self.params.rms_decay
    elif self.params.update_rule == "adam": # se invece è Adam allora c'è solo un parametro che è il learning rate
      d["lr"] = self.params.learning_rate
    return d

  def get_epsilon(self):
    """
    Scopo generale: Calcola l’epsilon corrente della strategia epsilon-greedy.
    Epsilon cambia nel tempo: di solito parte alto (molta esplorazione) e scende
    """
    #linear descent from 1 to 0.1 starting at the replay_start_time
    replay_start_time = max([self.frame_count-self.params.replay_start_size, 0]) # self.frame_count è il numero di step fatti finora, invece replay_start_size è quanti step aspetti prima di iniziare 
    # inizialmente la differenza è negativa quindi replay_start_time è 0
    epsilon = self.params.epsilon_start
    epsilon -= (self.params.epsilon_start - self.params.epsilon_min)*\
      (min(replay_start_time, self.params.epsilon_decay)/self.params.epsilon_decay) # questa è la formula per la discesa lineare
    return epsilon

  def get_mean_q_val(self, batch=1000):
    """
    Scopo generale: Misura “quanto grandi” sono i valori Q in media.
    È una metrica diagnostica: se Q esplode o collassa, puoi accorgertene.
    """
    imgs = self.exp_replay.random_batch(batch, random_selection=True) # imgs probabilmente contiene tuple (stato, azione, reward, next_state, done) o simile.
    """
    self.model.get_q_vals(imgs[0])
    = per ogni stato, la rete restituisce un vettore con Q per tutte le azioni.
    Forma tipica: (batch_size, num_actions).
    np.max(..., axis=1)
    = per ogni stato prende il massimo Q tra le azioni (cioè “miglior azione secondo il modello”).
    Forma: (batch_size,).
    np.mean(...)
    = fa la media su tutto il batch.
    Risultato: un singolo numero (Q medio).
    """
    return np.mean(np.max(self.model.get_q_vals(imgs[0]),axis=1))
    

  def run_testing(self, epoch):
    """
    valutare quanto è bravo l’agente senza allenarlo (o comunque in modalità “testing”).
    Questa funzione:
    fa giocare l’agente per un certo numero di step (steps_per_test)
    somma i reward
    calcola la media
    salva la media nei file CSV (con update_results)
    """
    total_reward = 0
    num_games = 0 # numero di episodi fatti
    original_frame_count = self.frame_count
    rem = self.params.steps_per_test
    while(self.frame_count - original_frame_count < self.params.steps_per_test): # continua fin quando il numero di step eseguiti nel test non è abbastanza 
      reward, fps = self.run_training_episode(self.max_frames_per_game, testing=True) # Chiama la funzione che gioca un episodio (si chiama “training episode” ma qui passa testing=True quindi è modalità test)
      print ("TESTING: %d fps,\t" % fps),
      print ("%d frames,\t" % self.ale.getEpisodeFrameNumber()),
      self.ale.reset_game() # reset ambiente atari alla fine dell'episodio 
      print "%d points,\t" % reward,
      rem = self.params.steps_per_test-(self.frame_count - original_frame_count) # Calcola quanti step di test mancano ancora.
      print "rem:", rem,
      print "ETA: %d:%02d" % (max(0, rem/60/fps*4), ((rem/fps*4)%60) if rem > 0 else 0),
      print "term ratio %.2f" % (100*self.term_ratio)
      total_reward += reward
      num_games += 1
    self.frame_count = original_frame_count # Ripristina il frame_count al valore pre-test: così il test non “consuma” step del training.
    mean_reward = round(float(total_reward)/num_games, 2)
    print "AVERAGE_SCORE:", mean_reward
    if type(self) is Q_Learning:
      """
      Qui fanno una cosa particolare:
      type(self) is Q_Learning → controlla se questa istanza è esattamente della classe Q_Learning (non un’altra classe).
      Se sì, allora prova a calcolare mean_q (valore Q medio) usando replay buffer.
      Ma solo se nn_file is None (cioè stai facendo training normale).
      """
      mean_q = self.get_mean_q_val() if self.params.nn_file is None else 1
    else:
      mean_q = 1
    self.update_results(epoch+1, mean_reward, mean_q)

  def train(self):
    """
    Questa funzione gestisce tutto il training:
    per ogni epoch:
    continua a giocare episodi finché non raggiunge steps_per_epoch
    stampa statistiche
    salva modello (best/last)
    salva statistiche termination
    fa una fase di test (run_testing)
    """
    cumulative_reward = 0
    counter = 0
    for i in range(self.params.epochs):
      start_frames = self.frame_count # Salva il frame_count all’inizio dell’epoch. Serve per sapere quanti step hai fatto “in questa epoch”.
      frames_rem = self.params.steps_per_epoch
      self.term_probs = [] # Lista vuota dove, durante questa epoch, verranno salvate termination probabilities
      while self.frame_count-start_frames < self.params.steps_per_epoch: # Continua finché non hai fatto steps_per_epoch step in questa epoch.
        total_reward, fps = self.run_training_episode(self.max_frames_per_game)
        cumulative_reward += total_reward
        frames_rem = self.params.steps_per_epoch-(self.frame_count-start_frames) # Aggiorna quanti frame mancano per completare l’epoch.
        print ("ep %d,\t") % (counter+1),
        print ("%d fps,\t" % fps),
        print ("%d frames,\t" % self.ale.getEpisodeFrameNumber()),
        self.ale.reset_game()
        print ('%d points,\t' % total_reward),
        print ('%.1f avg,\t' % (float(cumulative_reward)/(counter+1))),
        print "%d rem," % frames_rem, 'eps: %.4f' % self.get_epsilon(),
        print "ETA: %d:%02d" % (max(0, frames_rem/60/fps*4), ((frames_rem/fps*4)%60) if frames_rem > 0 else 0),
        print "term ratio %.2f" % (100*self.term_ratio)
        counter += 1

      if self.params.nn_file is None:
        self.save_model(total_reward)
      self.update_term_probs(i, self.term_probs)

      self.run_testing(i)

class DQN_Trainer(Trainer): # quindi questa classe eredita tutto quello che c'è nella classa Trainer 
  def __init__(self, **kwargs): # accetta un qualsiasi numero di argomenti e li mette in un dizionario
    super(DQN_Trainer, self).__init__(**kwargs) 
    """
    Questa riga vuol dire:
    “Chiama l’__init__ della classe padre (Trainer) passando gli stessi argomenti.”
    super(DQN_Trainer, self) = “il padre di DQN_Trainer”
    .__init__(...) = chiama il costruttore del padre
    (**kwargs) = passa tutti gli argomenti ricevuti
    """

  def run_training_episode(self, max_steps, testing=False):
    """
    Gioca una partita/episodio (o fino a max_steps) e durante il gioco:
    sceglie un’opzione o
    sceglie un’azione a usando l’opzione
    esegue l’azione nell’ambiente
    salva esperienza nel replay buffer
    aggiorna (learn actor + critic) se è training
    Alla fine ritorna:
    total_reward dell’episodio
    fps (velocità)
    """
    def get_new_frame(new_frame, x):
      
      new_x = np.empty((4, 84, 84), dtype="float32")
      new_x[0:3] = x[-3:]
      new_x[-1] = new_frame
      return new_x

    start_time = time.time()

    total_reward = 0
    data_set = self.test_replay if testing else self.exp_replay # sceglie dove salvare le transizioni: se testing=True, salva in test_replay altrimenti in exp_replay (experience replay del training)
    start_frame_count = self.frame_count
    x = self._init_ep()
    s = self.model.get_state([x]) # trasforma i frame x in uno stato s per la rete.
    game_over = self.ale.game_over() # controlla se il gioco è finito (all’inizio dovrebbe essere False).
    num_lives = self.ale.lives()
    current_option = 0 # inizializza opzione
    current_action = 0 # inizializza azione
    new_option = self.model.predict_move(s)[0] # chiede al modello di predire una “mossa”/opzione
    # In pratica: new_option è l’opzione proposta dal modello.
    termination = True
    episode_counter = 0
    termination_counter = 0 # per contare quante volte è avvenuta una terminazione ovvero quante volte si è cambiata l'opzione
    since_last_term = 1 # da quanti episodi sta continuando la stessa opzione
    while not game_over:
      self.frame_count += 1
      episode_counter += 1
      epsilon = self.get_epsilon() if not testing else self.params.optimal_eps
      if termination: # se l'opzione è terminata (quindi devi sceglierne una nuova)
        if self.print_option_stats:
          print "terminated -------", since_last_term,
        termination_counter += 1
        since_last_term = 1 # reserra da quanto dura l'opzione
        current_option = np.random.randint(self.params.num_options) if np.random.rand() < epsilon else new_option
        """
        Questa riga è cruciale. Significa:
        Con probabilità epsilon: scegli un’opzione random
        Altrimenti: scegli new_option proposta dal modello (policy-over-options)
        Quindi:
        np.random.rand() < epsilon → decide “esploro?”
        np.random.randint(self.params.num_options) → opzione random tra 0 e num_options-1

        QUESTO è EPSILON GREEDY MA SULLE OPZIONI
        """
        #current_option = self.get_option(epsilon, s)
      else: # se invece l'opzione non è terminata 
        if self.print_option_stats:
          print "keep going",
        since_last_term += 1 # incrementa la durata dell'opzione corrente 
      current_action = self.model.get_action(s, [current_option])[0]
      """
      chiede al modello: “dato lo stato s e l’opzione current_option, che azione faccio?”
      [0] perché batch size 1.
      Questa è la intra-option policy π(a|s,o)
      """
      #print current_option, current_action
      if self.print_option_stats:
        print current_option,# current_action
        #print [round(i, 2) for i in self.model.get_action_dist(s, [current_option])[0]]
        if True:
          self.action_counter[current_option][self.legal_actions[current_action]] += 1
          data_table = []
          option_count = []
          for ii, aa in enumerate(self.action_counter):
            s3 = sum([aa[a] for a in aa])
            if s3 < 1:
              continue
            print ii, aa, s3
            option_count.append(s3)
            print [str(float(aa[a])/s3)[:5] for a in aa]
            data_table.append([float(aa[a])/s3 for a in aa])
            print

          #ttt = self.model.get_action_dist(s3, [current_option])
          #print ttt, np.sum(-ttt*np.log(ttt))
          print

      reward, raw_reward, new_frame = self.act(current_action, testing=testing) # esegue l'azione

      game_over = self.ale.game_over() or (self.frame_count-start_frame_count) > max_steps
      new_num_lives = self.ale.lives()
      life_death = (new_num_lives < num_lives and not testing and self.params.death_ends_episode)
      num_lives = new_num_lives

      data_set.add_sample(x[-1], current_option, reward, game_over or life_death)
      # Questa riga salva nel replay: (frame corrente, opzione scelta, reward ottenuto, fine episodio), così il critic può allenarsi su molte esperienze passate

      old_s = copy.deepcopy(s)
      x = get_new_frame(new_frame, x)
      s = self.model.get_state([x]) 
      term_out = self.model.predict_termination(s, [current_option]) # qui bisogna vedere cosa ritorna predict_termination. (immagino la funzione Beta dia una probabilità di terminazione)
      termination, new_option = term_out[0][0], term_out[1][0]
      if self.frame_count < self.params.replay_start_size and not testing:
        termination = 1
        """
        Se sei ancora nella fase iniziale (prima di iniziare ad allenare), forzano termination=1.
        Significa: cambia opzione ad ogni step (o molto spesso).
        Serve a esplorare di più opzioni prima che il training inizi.
        """
      total_reward += raw_reward
      if self.frame_count > self.params.replay_start_size and not testing: # inzia ad imprare solo dopo aver riempito abbastanza replay buffer e se non è in testing
        self.learn_actor(old_s,
                         np.array(x).reshape(1,4,84,84),
                         [current_option],
                         [current_action],
                         [reward],
                         [game_over or life_death])
        if self.frame_count % self.params.update_frequency == 0:
          self.learn_critic() # ogni update_frequency step aggiorna il critic (valori Q).
        if self.frame_count % self.params.freeze_interval == 0:
          if self.params.freeze_interval > 999:
            print "updated_params"
          self.model.update_target_params()
          # ogni freeze_interval step aggiorna i parametri della target network.

    #print self.last_dist
    self.term_ratio = float(termination_counter)/float(episode_counter) # calcola frazione di step in cui c’è stata una terminazione (o quante terminazioni rispetto agli step).
    if not testing:
      self.term_probs.append(self.term_ratio)
    if self.print_option_stats:
      print "---->", self.term_ratio
      #self.print_table(data_table, option_count)
    fps = round((self.frame_count - start_frame_count)/(time.time()-start_time), 2)
    fps = self.ale.getEpisodeFrameNumber()/(time.time()-start_time)
    return total_reward, fps

  def print_table(self, conf_arr, d1):
    pickle.dump(np.array(conf_arr), open( "/".join(self.params.nn_file.split("/")[:-1])+"/confu_data.pkl", "wb" ) )
    self.plot_table(np.array(conf_arr), d1)
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    res = ax.imshow(np.array(norm_conf).T, cmap=plt.cm.jet, 
                    interpolation='nearest', vmax=1, vmin=0)

    width, height = conf_arr.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(conf_arr[x][y]*100)), xy=(x, y), 
                        horizontalalignment='center',
                        verticalalignment='center', size=9)

    cb = fig.colorbar(res)
    plt.xticks(range(width), range(1,1+width))
    plt.yticks(range(height), [self.legal_actions[iii] for iii in range(height)])
    plt.savefig("/".join(self.params.nn_file.split("/")[:-1])+"/"+self.params.nn_file.split("/")[-2].replace(".", "_")+'_confu.png', bbox_inches='tight', format='png')
    raise NotImplemented

  def learn_actor(self, s, next_x, o, a, r, term):
    """
    Scopo generale:
    In un algoritmo Actor-Critic / Option-Critic, l’actor è la parte che decide “cosa fare”:
    nel tuo caso: decide l’azione dentro un’opzione (intra-option policy) e spesso anche influenza la termination β(s,o).
    Questa funzione serve a fare un passo di aggiornamento (training) dell’actor usando la transizione che hai appena osservato.
    """
    td_errors = self.model.train_conv_net(s, next_x, o, r, term, actions=a, model="actor")
    # Questa riga chiama la funzione di training della rete (train_conv_net) dicendole di aggiornare la parte “actor” (model="actor"), usando la transizione 
    # appena osservata: stato s, prossima osservazione next_x, opzione o, azione a, reward r e flag di fine episodio term.
    # La funzione ritorna dei TD errors, cioè un segnale di quanto la stima del valore/target era sbagliata, utile per l’update dell’actor.
    return td_errors

  def learn_critic(self):
    x, o, r, next_x, term = self.exp_replay.random_batch(self.params.batch_size) # campiona un batch dal replay
    td_errors = self.model.train_conv_net(x, next_x, o, r, term, model="critic")
    return td_errors

class Q_Learning(DQN_Trainer): # questa classe eredita da DQN_Trainer che a sua volta eredita da Trainer
  def __init__(self, **kwargs):
    super(Q_Learning, self).__init__(**kwargs)
    """
    Nel gioco Atari ho delle immmagini di gioco e quindi uso una CNN per analizzarle
    mlp si intende un layer fully connected
    """
    model_network = [{"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4],
                     "out_size": 32, "activation": "relu"},
                     {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2],
                     "out_size": 64, "activation": "relu"},
                     {"model_type": "conv", "filter_size": [3,3], "pool": [1,1], "stride": [1,1],
                     "out_size": 64, "activation": "relu"},
                     {"model_type": "mlp", "out_size": 512, "activation": "relu"},
                     {"model_type": "mlp", "out_size": len(self.legal_actions), "activation": "linear"}]

    learning_params = self.get_learning_params()

    self.model = OptionCritic_Network(model_network=model_network,
      learning_method=self.params.update_rule, dnn_type=self.params.USE_DNN_TYPE, clip_delta=self.params.clip_delta,
      input_size=[None,4,84,84], batch_size=self.params.batch_size, learning_params=learning_params,
      gamma=self.params.discount, freeze_interval=self.params.freeze_interval,
      termination_reg=self.params.termination_reg, num_options=self.params.num_options,
      actor_lr=self.params.actor_lr, double_q=self.params.double_q, temp=self.params.temp,
      entropy_reg=self.params.entropy_reg, BASELINE=self.params.baseline)
    if self.params.nn_file is not None:
      self.model.load_params(pkl.load(open(self.params.nn_file, 'r')))

    self.exp_replay = DataSet(84, 84, self.rng, max_steps=self.params.replay_memory_size, phi_length=4)
    self.test_replay = DataSet(84, 84, self.rng, max_steps=4, phi_length=4)

if __name__ == "__main__":
  pass
