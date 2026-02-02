import re
import os
import cmd
import sys
import time
import msvcrt
import traceback
import numpy  as np
import psutil as pu
import pandas as pd # used to support some external scripts

from os           import makedirs
from os.path      import join, exists, basename
from collections  import defaultdict, Counter
from math         import floor, ceil
from itertools    import chain
from primefac     import primefac
from scipy.linalg import circulant, qr

from sklearn.tree import export_text, export_graphviz
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from customDefs   import getMountedOn, setupEssayConfig, getEssayParameter
from customDefs   import PrintBuffer, serialise, deserialise
from customDefs   import saveAsText, loadAsText

from datasets     import load_dataset, assign_groups
from datasets     import list_conflicts, reverse_conflicts, remove_conflicts
from datasets     import get_labels, split_dataset
from datasets     import presence2rank, rank2presence, membership2presence
from datasets     import count_labels, report_split_stats
from metrics      import f1_micro, f1_macro, f1_weigh, hammingl, mse, ktau
from evaluation   import get_model, evaluate, estimate, diagnose, get_tikz_diag
from polygrid     import build_Polygrid_config
from layout       import layout_data
from logo         import show_Polygrid_logo

from customDefs   import ECO_SEED
from customDefs   import ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE
from customDefs   import ECO_HIT, ECO_MISS
from customDefs   import ECO_THRSHLDLVLS
from datasets     import ECO_DB_UNLABELLED, ECO_DB_MULTICLASS
from datasets     import ECO_DB_MULTILABEL, ECO_DB_LABELRANK
from datasets     import ECO_ASSIGNMENT_O, ECO_ASSIGNMENT_H
from datasets     import ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R
from datasets     import ECO_SPLIT_MVS, ECO_SPLIT_STV
from polygrid     import ECO_RCOND

# constants that indicate allowable internal states of the CLI
ECO_VOID     = 0
ECO_LOADED   = 1
ECO_ASSIGNED = 2
ECO_SPLIT    = 3
ECO_TRAINED  = 4
ECO_FITTED   = 5

state2txt = {
  ECO_VOID     : 'void',
  ECO_LOADED   : 'loaded',    # attributes and factors have been successfully loaded
  ECO_ASSIGNED : 'assigned',  # targets have been successfully loaded/created
  ECO_SPLIT    : 'split',     # dataset successfully split into training/test partitions
  ECO_TRAINED  : 'trained',   # a Polygrid model has been trained
  ECO_FITTED   : 'fitted',    # a competing model has been fitted
  }

# character and regex used to separate commands in a single-line script
# -- used in PolygridCLI.do_run(...) and PolygridCLI.API(...)
ECO_INSTR_SEP = ';'
ECO_INSTR_SPLITTER = r'''((?:[^;"']|"[^"]*"|'[^']*')+)'''

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def rank_key_by_value(L, reverse=False, clip_value=True, decimals=3):
  # assumes L ~ [(key, value), ...]
  # if informed, val_format must be compatible to float
  sign = -1 if reverse else 1
  L1 = sorted(L, key=lambda e: sign * e[1])
  if(clip_value):
    L2 = [k for (k,v) in L1]
  else:
    L2 = [(k, np.round(v, decimals=decimals)) for (k,v) in L1]
  return L2

# for specs of cmd.Cmd, see https://docs.python.org/3/library/cmd.html
class PolygridCLI(cmd.Cmd):

  prompt = '(Poly) '
  intro  = 'Welcome to Polygrid command-line interface. Type in "help" for a list of commands.\n'

  def __init__(self, completekey='tab', stdin=None, stdout=None, configfile=None):

    super().__init__(completekey, stdin, stdout)

    # list of attributes set to values from the config file
    self.configfile   = configfile
    self.essayruns    = None
    self.sourcepath   = None
    self.targetpath   = None
    self.offlineval   = None
    self.polygrid     = None
    self.maxlabels    = None
    self.testfrac     = None
    self.balance      = None
    self.hide_tags    = None
    self.maxcores     = None

    # list of attributes set to default values and redefined during running time
    # (CLI workarea)
    self.cmdcounter   = 0        # command counter, updated in self.precmd(...)
    self.exccounter   = 0        # exception counter, updated in self.print_exc(...)
    self.snapshots    = []       # performance data of models evaluated during the session
    self.state        = ECO_VOID # record of the current environment' state machine
    self.topN         = 6
    self.gridCols     = 7
    self.scaleCols    = 1
    self.transpose    = False
    self.fuzzfactor   = 1.0
    self.alpha        = 0.05     # used in bootstrapped confidence intervals
    self.rcond        = 1E-2     # cutoff for small singular values in np.linalg.lstsq(..., rcond)
    self.verbose      = True
    self.showprotos   = True
    self.usecaseids   = True
    self.cutoff       = ECO_CUTOFF_MULTIPLE
    self.metric       = f1_micro
    self.avoid_FP     = False # if True, avoids FP when learning_thresholds, and FN otherwise
    self.corder       = 'original'
    self.register     = {}   # user can store objects here during CL interaction
    self.buffer       = {}   # stores the content recovered by command 'input'
    self.column2label = None # stores a live, user-defined domain label translator

    # list of attributes defined during running time, related to LASTE pipeline
    # (LASTE workarea - Load, Assign, Split, Train, and Evaluate cycle)
    self.class_noun   = 'Label' # noun used when original assignments are replaced
    self.dataset      = None # dataset data; sklearn-like Bunch object
    self.assignmethod = None # assignment metadata
    self.assignpars   = None # assignment metadata
    self.splitmethod  = None # split metadata
    self.splitpars    = None # split metadata
    self.splitdesc    = None # split metadata
    self.splitmsg     = None # split metadata
    self.model_type   = None # model metadata; the type of model currently active
    self.model_config = None # model metadata; the config used to instantiate the model
    self.model        = None # model data; a trained model instance (Polygrid, competitor)
    self.model_size   = None # model metadata; the number of weights of the last trained Polygrid model
    self.model_hist   = []   # model history (list with model sizes; only used in do_assess)

    self.X            = None # workarea (for load, factor)
    self.Y            = None # workarea (for load, factor, assign)
    self.U            = None # workarea (for load, factor, assign)
    self.te_idxs      = None # workarea (for split)
    self.tr_idxs      = None # workarea (for split)
    self.Y_real       = None # workarea (for train)
    self.Y_pred       = None # workarea (for train)
    self.YorU_hat     = None # workarea (for train)
    self.performance  = None # workarea (for train)
    self.performstat  = None # workarea (for train)
    self.inspection   = None # workarea (for inspect)
    self.measures     = None # workarea (for range & search)
    self.filename     = None # workarea (for show)

    # data recovered from an offline evaluation
    self.offlineval_pending_upd = False
    self.offlineval_datasets    = None
    self.offlineval_datapars    = None
    self.offlineval_competitors = None
    self.offlineval_results     = None
    self.offlineval_configs     = None

    # backup copies of class-related attributes of the dataset used in assignment
    self.bk_atype        = None
    self.bk_class_noun   = None
    self.bk_target_names = None
    self.last_nspd_na    = None

  def reset(self, state):

    self.state = state

    if(self.state == ECO_VOID):
      self.dataset      = None
      self.assignmethod = None
      self.assignpars   = None
      self.splitmethod  = None
      self.splitpars    = None
      self.splitdesc    = None
      self.splitmsg     = None
      self.model_type   = None
      self.model_config = None
      self.model        = None
      self.model_size   = None

      self.X            = None
      self.Y            = None
      self.U            = None
      self.te_idxs      = None
      self.tr_idxs      = None
      self.Y_real       = None
      self.Y_pred       = None
      self.YorU_hat     = None
      self.performance  = None
      self.performstat  = None
      self.inspection   = None
      self.measures     = None
      self.filename     = None

      self.bk_atype        = None
      self.bk_class_noun   = None
      self.bk_target_names = None

    elif(self.state == ECO_LOADED):

      if(self.bk_atype is None):
        self.bk_atype        = self.dataset.atype
        self.bk_class_noun   = self.dataset.class_noun
        self.bk_target_names = self.dataset.target_names
      self.X = self.dataset.data
      self.Y = self.dataset.target

      self.assignmethod = None
      self.assignpars   = None
      self.splitmethod  = None
      self.splitpars    = None
      self.splitdesc    = None
      self.splitmsg     = None
      self.model_type   = None
      self.model_config = None
      self.model        = None
      self.model_size   = None

      self.U            = None
      self.te_idxs      = None
      self.tr_idxs      = None
      self.Y_real       = None
      self.Y_pred       = None
      self.YorU_hat     = None
      self.performance  = None
      self.performstat  = None
      self.inspection   = None
      self.measures     = None
      self.filename     = None

    elif(self.state == ECO_ASSIGNED):

      self.splitmethod  = None
      self.splitpars    = None
      self.splitdesc    = None
      self.splitmsg     = None
      self.model_type   = None
      self.model_config = None
      self.model        = None
      self.model_size   = None

      self.te_idxs      = None
      self.tr_idxs      = None
      self.Y_real       = None
      self.Y_pred       = None
      self.YorU_hat     = None
      self.performance  = None
      self.performstat  = None
      self.inspection   = None
      self.measures     = None
      self.filename     = None

    elif(self.state == ECO_SPLIT):

      self.model_type   = None
      self.model_config = None
      self.model        = None
      self.model_size   = None

      self.Y_real       = None
      self.Y_pred       = None
      self.YorU_hat     = None
      self.performance  = None
      self.performstat  = None
      self.inspection   = None
      self.measures     = None
      self.filename     = None

    elif(self.state in [ECO_TRAINED, ECO_FITTED]):

      self.inspection   = None
      self.measures     = None
      self.filename     = None

  def _get_data_key(self):
    if(self.state < ECO_TRAINED):
      return None
    (m,n) = self.Y.shape
    key = self.dataset.name + self.assignmethod + str(n) + self.splitdesc
    return hash(key)

  def _get_model_key(self):
    if(self.state < ECO_TRAINED):
      return None
    key = self.model_type + str(self.model_config) + str(self.model_size)
    return hash(key)

  def random_state(self):
    return ECO_SEED + self.cmdcounter

  def increase_exc(self):
    self.exccounter += 1

  def print_exc(self):
    self.increase_exc()
    traceback.print_exc()

  #-------------------------------------------------------------------------------------
  # specifies administrative functions
  #-------------------------------------------------------------------------------------
  def preloop(self, static_logo=False):

    setupEssayConfig(self.configfile)
    self.essayruns   = getEssayParameter('ESSAY_RUNS')
    self.sourcepath  = getEssayParameter('PARAM_SOURCEPATH')
    self.targetpath  = getEssayParameter('PARAM_TARGETPATH')
    self.offlineval  = getEssayParameter('PARAM_FILENAME')
    self.polygrid    = getEssayParameter('PARAM_POLYGRID')
    self.maxlabels   = getEssayParameter('PARAM_MAXLABELS')
    self.testfrac    = getEssayParameter('PARAM_TESTFRAC')
    self.balance     = getEssayParameter('PARAM_BALANCE')
    self.hide_tags   = getEssayParameter('PARAM_HIDE_TAGS')
    self.maxcores    = getEssayParameter('PARAM_MAXCORES')

    show_Polygrid_logo(self.configfile, static_logo=True)

  def precmd(self, line):
    self.cmdcounter += 1
    try:
      if(len(line) == 0):
        line = 'pass'
      elif(line[0] == '#'):
        line = 'pass'
      else:
        args = line.split()
        args[0] = args[0].lower()
        line = ' '.join(args)
      return line
    except:
      return line

  def do_quit(self, line):
    'Quits the Polygrid CLI'
    if(self.offlineval_pending_upd):
      pass #self.do_update('')

    if(self.verbose):
      print(f'Quitting session with {self.exccounter} exception(s).')
    return True

  def do_hook(self, line):
    """Forces Polygrid CLI to enter debug mode, e.g.,

    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- hook

       (enters in debug mode)
       --Return--
       > .../code/cli.py(378)do_hook()->None
       -> breakpoint()

       (Pdb) b polygrid:2703
       (Pdb) cont
       (returns to CLI)

    -- show scales -50
       (enters in debug mode)
       > .../code/polygrid.py(2703)show_scales()
       -> (Y_pred, YorU_hat, names, coords, cutoffs, yticklabels) = self.inspect(X, Y, idxs, tr_idxs)

    """
    breakpoint()

  def do_eof(self, line):
    'Quits the Polygrid CLI'
    return self.do_quit(line)

  def do_pass(self, line):
    'Does nothing -- used with input command files'

  def do_p(self, line):
    "Evaluates an expression (in Python), e.g., \n-- p self.Y_real"
    try:
      res = eval(line)
      print(res)
    except Exception as e:
      self.print_exc()

  def do_exec(self, line):
    "Executes an expression (in Python), e.g., \n-- exec self.register['L'] = Counter([str(e) for e in self.Y])"
    try:
      res = exec(line)
    except Exception as e:
      self.print_exc()

  def do_clear(self, line):
    'Clears the terminal'
    try:
      res = os.system('cls' if os.name == 'nt' else 'clear')
    except Exception as e:
      self.print_exc()

  def do_about(self, line):
    'Shows information about the Polygrid CLI'
    try:
      show_Polygrid_logo(self.configfile, static_logo=False)
    except Exception as e:
      self.print_exc()

  def do_input(self, line):
    """Allows for user input, e.g.,
    -- input Enter the number of sectors per domain:
    -- p self.buffer
    """
    self.buffer = input(line)

  def do_pause(self, line):
    'Waits until ENTER is pressed'
    print('Press ENTER to continue')
    c = msvcrt.getch()
    while(c != b'\r'):
      c = msvcrt.getch()

  def do_ll(self, line, show_descr=True):
    """Shows the details about the internal state of the interface, e.g.,

    Shows details of internal state, with short database description
    -- ls
    -- ls % (counts of labels per partition shown as percentages of total)
    -- ls -tr (switches the performance report to be based on training data)

    Shows details of internal state, with long database description
    -- ll
    -- ll % (counts of labels per partition shown as percentages of total)
    -- ll -tr (switches the performance report to be based on training data)
    """
    try:
      print(f'Process ID ...: {os.getpid()}')
      print(f'Memory usage .: {pu.Process().memory_info().rss/2**20:.1f} MB')
      print(f'Current state : {state2txt[self.state]}')
      print()
      print(f'Dataset ......: {None if self.dataset is None else self.dataset.name}')
      self._desc_dataset(line, show_descr=show_descr)
      print(f'Assignment ...: {self.assignmethod}')
      self._desc_assignments(line)
      if(self.state < ECO_TRAINED):
        print(f'Model ........: {self.model_type}')
      else:
        if(self.model.scenario in [ECO_DB_MULTICLASS]):
          print(f'Model ........: {self.model.model_name}, with {self.model.get_size()} weights')
        else:
          if(self.model.cutoff == ECO_CUTOFF_SINGLE):
            print(f'Model ........: {self.model.model_name}, with {self.model.get_size()} weights, 1 cutoff')
          else:
            print(f'Model ........: {self.model.model_name}, with {self.model.get_size()} weights, {self.model.n} cutoffs')

      if('-tr' in line or self.performstat == 'training'):
        print('-------------------------------')
        print('--- PERFORMANCE IN TRAINING ---')
        print('-------------------------------')
      self._desc_performance(line)

    except Exception as e:
      self.print_exc()

  def do_ls(self, line, show_descr=False):
    """Shows the details about the internal state of the interface, e.g.,

    Shows details of internal state, with short database description
    -- ls
    -- ls % (counts of labels per partition shown as percentages of total)
    -- ls -tr (switches the performance report to be based on training data)

    Shows details of internal state, with long database description
    -- ll
    -- ll % (counts of labels per partition shown as percentages of total)
    -- ll -tr (switches the performance report to be based on training data)
    """
    self.do_ll(line, show_descr=show_descr)

  def _desc_dataset(self, line, show_descr=False): #xxx make this signature look like _desc_assigments

    if(self.state < ECO_LOADED):
      pass
    else:
      if(show_descr):
        print('-- {0}'.format(self.dataset.DESCR))
      (m,d) = self.X.shape
      print('-- {0} is being handled as a(n) {1} dataset'.format(self.dataset.name, self.dataset.atype))
      print('-- {0} has {1} instances, {2} attributes'.format(self.dataset.name, m, d))
      print('-- attributes: {0}'.format(', '.join(self.dataset.feature_names)))
      if(self.dataset.atype in [ECO_DB_UNLABELLED]):
        print('-- targets : this dataset has no labels.')
      else:
        aux = [f'({i}) {e}' for (i,e) in enumerate(self.dataset.target_names)]
        print('-- targets : {0}'.format(', '.join(aux)))
      if(hasattr(self.dataset, 'factors') and len(self.dataset.factor_names) > 0):
        print('-- factors : {0}'.format(', '.join(self.dataset.factor_names)))
      else:
        print('-- factors : this dataset has no factors')

    return None

  def _get_evaluation_scenario(self): #xxx maybe _get_task_type better?
    if(self.assignmethod == ECO_ASSIGNMENT_O):
      scenario = self.dataset.atype
    elif(self.assignmethod == ECO_ASSIGNMENT_H):
      scenario = ECO_DB_MULTICLASS
    elif(self.assignmethod == ECO_ASSIGNMENT_F):
      scenario = ECO_DB_MULTILABEL
    elif(self.assignmethod == ECO_ASSIGNMENT_R):
      scenario = ECO_DB_LABELRANK
    else:
      scenario = None
    return scenario

  def _desc_assignments(self, line, verbose=True):

    def _desc_partition(partition, mask, Y, g, show_percents=False):
      (m,n) = Y.shape
      s = Y.sum()
      content = []
      content.append((mask + ' has {1:5d} instances, {2:3d} labels, ').format(partition, m, len(g)))
      for label in sorted(g):
        if(show_percents):
          content.append(f'{label}: {g[label]/s:4.2f}')
        else:
          content.append(f'{label}: {g[label]:3d}')
      return content

    if(self.state < ECO_ASSIGNED):
      content = None

    else:
      args = line.strip().split()
      show_percents = ('%' in args)

      # computes and displays the distribution of labels in different dataset partitions
      views = [('dataset',    '-- {0:<8}', self.Y, 1),]
      if(self.state >= ECO_SPLIT):
        views.append(('training', '\n-- {0:<8}', self.Y[self.tr_idxs,:], -1))
        views.append(('test',     '\n-- {0:<8}', self.Y[self.te_idxs,:], -1))

      # computes relevant statistics of the dataset partition
      (m,n) = self.Y.shape
      total = np.zeros(n, dtype=int)
      content = []
      g = {}
      scenario = self._get_evaluation_scenario()
      for (partition, mask, Y, sign) in views:
        g[partition] = count_labels(Y, scenario)
        content += _desc_partition(partition, mask, Y, g[partition], show_percents)
        label_counts = np.array([g[partition][label] for label in sorted(g[partition])])
        total += sign * np.array(label_counts)
        content += report_split_stats(Y, scenario)

      if(self.state >= ECO_SPLIT):
        (partition, mask) = ('balance', '\n-- {0:<8}')
        g[partition] = {i: total[i] for i in range(n)}
        content += _desc_partition(partition, mask, self.Y, g[partition], show_percents)

        # if split did not attain target threshold, saves a message with details
        if(self.splitmsg is not None):
          content.append(f'\n-- split {self.splitmsg}')

      content = ' '.join(content)
      if(verbose):
        print(content)

    return content

  def _desc_performance(self, line):

    if(self.state < ECO_TRAINED):
      pass
    else:

      if('-tr' in line):
        # overrides the current performance evaluation with statistics
        # based on the cases in the training partition
        scenario = self._get_evaluation_scenario()
        Y_real = self.Y[self.tr_idxs]
        (Y_pred, YorU_hat) = self.model.predict(self.X[self.tr_idxs], return_scores=True)
        (res, report) = evaluate(Y_real, Y_pred, YorU_hat, scenario)
        self.performance = (res, report)
        self.performstat = 'training'

      # recovers the current performance evaluation and prints it
      (res, report) = self.performance
      print(report)

    return None


  #-------------------------------------------------------------------------------------
  # specifies commands that implement the LASTE cycle (Load Assign Split Train Evaluate)
  #-------------------------------------------------------------------------------------

  def do_load(self, line):
    """Loads a dataset, e.g.,

    Multiclass datasets from the UCI repo (via sklearn.datasets API):
    -- load iris
    -- load wine
    -- load cancer (only the attributes related to mean measurements are loaded)
    -- load digits

    Multiclass datasets from other repositories (Palmer's penguins):
    -- load penguins

    Multilabel datasets from Cometa and KDIS repositories:
    -- load foodtruck (nominal attributes loaded as factors, ignored in training)
    -- load water

    Label ranking datasets from the Paderborn repo:
    -- load iris@pb
    -- load wine@pb
    -- load vowel@pb

    Healthcare-related datasets (Brazil):
    -- load whoqol       (multiclass)
    -- load whoqol-ml-11 (multilabel)
    -- load whoqol-ml-22 (multilabel)
    -- load whoqol-lr-11 (label ranking)
    -- load whoqol-lr-22 (label ranking)

    -- load ampiab       (multiclass)
    -- load ampiab-ml-11 (multilabel)
    -- load ampiab-ml-22 (multilabel)
    -- load ampiab-lr-11 (label ranking)
    -- load ampiab-lr-22 (label ranking)

    -- load elsio1       (multiclass)
    -- load elsio1-ml-11 (multilabel)
    -- load elsio1-ml-22 (multilabel)
    -- load elsio1-lr-11 (label ranking)
    -- load elsio1-lr-22 (label ranking)
    """

    try:
      args = line.strip().split()
      datasetname = args[0]

      self.reset(ECO_VOID)
      self.dataset = load_dataset(self.sourcepath, datasetname)
      self.reset(ECO_LOADED)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_VOID)

  def do_factor(self, line):
    """Selects instances whose factors match a given regular expression, e.g.,

    -- load penguins
       penguins has 333 instances, 4 attributes
       factors : island, sex, year

    -- factor /biscoe
       selected rows with factors ['island', 'sex', 'year'] ~ /biscoe
       penguins dataset has 163 instances, 4 attributes

    -- factor .*/female
       selected rows with factors ['island', 'sex', 'year'] ~ .*/female
       penguins dataset has 165 instances, 4 attributes

    -- factor .*/2007    # select speciments whose data were collected in 2007
       selected rows with factors ['island', 'sex', 'year'] ~ .*/2007
       penguins dataset has 103 instances, 4 attributes

    -- factor /          # restores the original dataset content
       penguins dataset has 333 instances, 4 attributes
       attributes: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

    Factors can be combined, e.g.

    -- factor .*/female/2009        # select female specimens assessed in 2009
    -- factor (?!/biscoe).*/female  # select female specimens from all islands but Biscoe

    To list the factor combinations, use:

    -- p sorted(set(self.dataset.factors))
    """

    if(self.state < ECO_LOADED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    if(not hasattr(self.dataset, 'factors')):
      print('-- factors : this dataset has no factors')
      return False

    try:

      if(len(line) > 0):
        # selects only dataset rows with factors that match the regular expression
        pattern = line
        regex = re.compile(pattern)
        f_row_filter = np.vectorize(lambda factor_path: bool(regex.match(factor_path)))
        idxs_factors = f_row_filter(self.dataset.factors)
        if(self.dataset.xdata is None):
          self.dataset.xdata    = self.dataset.data
          self.dataset.xodata   = self.dataset.odata
          self.dataset.xtarget  = self.dataset.target
          self.dataset.xcaseIDs = self.dataset.caseIDs
        self.dataset.data    = self.dataset.xdata[idxs_factors]
        self.dataset.odata   = self.dataset.xodata[idxs_factors]
        self.dataset.target  = self.dataset.xtarget[idxs_factors]
        self.dataset.caseIDs = list(np.array(self.dataset.xcaseIDs)[idxs_factors])
        factor_names = self.dataset.factor_names
        print('-- selected rows with factors {0} ~ {1}'.format(factor_names, pattern))

      else:
        # restores the dataset to its original content (before filtering)
        if(self.dataset.xdata is not None):
          self.dataset.data     = self.dataset.xdata
          self.dataset.odata    = self.dataset.xodata
          self.dataset.target   = self.dataset.xtarget
          self.dataset.caseIDs  = self.dataset.xcaseIDs
          self.dataset.xdata    = None
          self.dataset.xodata   = None
          self.dataset.xtarget  = None
          self.dataset.xcaseIDs = None

      self.reset(ECO_LOADED)

      (m,d) = self.dataset.data.shape
      print('-- {0} dataset has {1} instances, {2} attributes'.format(self.dataset.name, m, d))
      print('-- attributes: {0}'.format(', '.join(self.dataset.feature_names)))

    except Exception as e:
      self.print_exc()
      self.reset(ECO_VOID)

  def do_upgrade(self, line):
    """Upgrades the task of the current dataset, e.g.

    Changes the task from multiclass classification to multilabel classification:
    -- load iris
    -- assign o
    -- upgrade
       upgraded from multiclass to multilabel
    -- split
    -- train 1 1
    -- config
       Summary of current context and model configuration
       dataset  iris
       task     multilabel
       model    Polygrid, with 12 weights
       vorder   averages
       annuli   1, s-invariant
       sectors  4, cover
       solver   lstsq
       cutoff   multiple

    Changes the task from multilabel classification to label ranking:
    -- load foodtruck
    -- assign o
    -- upgrade
       upgraded from multilabel to label ranking
    -- split
    -- train 1 1
    -- config
       Summary of current context and model configuration
       dataset  foodtruck
       task     label ranking
       model    Polygrid, with 132 weights
       vorder   averages
       annuli   1, s-invariant
       sectors  11, cover
       solver   lstsq
       cutoff   multiple
    """

    if(self.state < ECO_ASSIGNED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:

      if(self.dataset.atype == ECO_DB_MULTICLASS):
        self.dataset.atype = ECO_DB_MULTILABEL
        self.assignmethod  = ECO_ASSIGNMENT_F
        print(f'-- upgraded from multiclass to multilabel')

      elif(self.dataset.atype == ECO_DB_MULTILABEL):
        self.dataset.atype = ECO_DB_LABELRANK
        self.assignmethod  = ECO_ASSIGNMENT_R
        self.Y = membership2presence(self.U)
        print(f'-- upgraded from multilabel to label ranking')

      else:
        raise ValueError(f'-- No generalisation of {self.dataset.atype} tasks has been implemented.')

      if(self.state >= ECO_TRAINED):
        self.reset(ECO_SPLIT)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_VOID)

  def do_downgrade(self, line):
    """Downgrades the task of the current dataset, e.g.,

    Changes the task from label ranking to multilabel classification.
    -- load iris@pb
    -- assign o
    -- downgrade
       downgraded from label ranking to multilabel
    -- split
    -- train 1 1
    -- config
       Summary of current context and model configuration
       dataset  iris@pb
       task     multilabel
       model    Polygrid, with 12 weights
       vorder   averages
       annuli   1, s-invariant
       sectors  4, cover
       solver   lstsq
       cutoff   multiple

    Changes the task from multilabel classification to multiclass classification.
    -- load foodtruck
    -- assign o
    -- downgrade
       downgraded from multilabel to multiclass
    -- split
    -- train 1 1
    -- config
       Summary of current context and model configuration
       dataset  foodtruck
       task     multiclass
       model    Polygrid, with 132 weights
       vorder   averages
       annuli   1, s-invariant
       sectors  11, cover
       solver   lstsq
    """

    if(self.state < ECO_ASSIGNED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:

      if(self.dataset.atype == ECO_DB_LABELRANK):
        self.dataset.atype = ECO_DB_MULTILABEL
        self.assignmethod  = ECO_ASSIGNMENT_F
        self.Y = rank2presence(self.Y)
        print(f'-- downgraded from label ranking to multilabel')

      elif(self.dataset.atype == ECO_DB_MULTILABEL):
        self.dataset.atype = ECO_DB_MULTICLASS
        self.assignmethod  = ECO_ASSIGNMENT_H
        self.Y = (self.U == self.U.max(axis=1)[:,None]).astype(int)
        if((self.Y.sum(axis=1) != 1).sum() > 0):
          # the original dataset is multilabel, so matrix U does not induce rankings
          # in this case, we take the first active label using the natural label order
          (m,n) = self.Y.shape
          Y = np.zeros((m,n), dtype=int)
          assignments = (self.U == self.U.max(axis=1)[:,None]).astype(int).argmax(axis=1)
          for i in range(m):
            j = assignments[i]
            Y[i,j] = 1

        print(f'-- downgraded from multilabel to multiclass')

      else:
        raise ValueError(f'-- No generalisation of {self.dataset.atype} tasks has been implemented.')

      if(self.state >= ECO_TRAINED):
        self.reset(ECO_SPLIT)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_VOID)

  def do_conflicts(self, line):
    """Lists, reverses, or removes attributes that negatively correlate with others, e.g., 

    Lists the attributes that negatively correlates with others, ranked by ocurrences
    -- conflicts

    Reverses the attributes that negatively correlates with others
    -- conflicts -re

    Removes the attributes that negatively correlates with others
    -- conflicts -rm
    """

    if(self.state < ECO_LOADED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:

      args = line.strip().split()
      if(len(args) == 0):
        # lists the conflicts in the description matrix
        content = list_conflicts(self.dataset.data, self.dataset.feature_names)
        print()
        print('\n'.join([f'{field}\t\t{count}' for (field, count) in content]))
        print()

      elif(args[0] == '-re'):
        # reverses the conflicting attributes
        (self.dataset.data, self.dataset.feature_names) = reverse_conflicts(self.dataset.data, self.dataset.feature_names)
        self.reset(ECO_LOADED)
        if(self.verbose):
          print('done')

      elif(args[0] == '-rm'):
        # removes the conflicting attributes
        (self.dataset.data, self.dataset.feature_names) = remove_conflicts(self.dataset.data, self.dataset.feature_names)
        self.reset(ECO_LOADED)
        if(self.verbose):
          print('done')

      else:
        raise ValueError(f'Invalid switch: {args[0]}')

    except Exception as e:
      self.print_exc()
      self.reset(ECO_VOID)

  def do_corr(self, line):
    'Displays (and saves) the correlation matrix of the description matrix'

    try:
      args = line.strip().split()

      if('-tr' in args):

        if(self.state < ECO_SPLIT):
          print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
          return False

        (m,d) = self.dataset.data[self.tr_idxs,:].shape
        C = np.corrcoef(self.dataset.data[self.tr_idxs,:], rowvar=False)
        np.savetxt('cormat_{0}.csv'.format(self.dataset.name), C)

      else:

        if(self.state < ECO_LOADED):
          print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
          return False

        (m,d) = self.dataset.data.shape
        C = np.corrcoef(self.dataset.data, rowvar=False)
        np.savetxt('cormat_{0}.csv'.format(self.dataset.name), C)

      print()
      print('-- {0} dataset has {1} instances, {2} attributes'.format(self.dataset.name, m, d))
      print('-- attributes: {0}'.format(', '.join(self.dataset.feature_names)))
      print(C)

    except Exception as e:
      self.print_exc()

  def do_assign(self, line):
    """Assign instances to labels, e.g.,

    Accept original labels from the dataset
    --------------------------------------------------------------------------------------
    -- assign o

    Assign labels using hierarchical clustering (to create a multiclass dataset)
    --------------------------------------------------------------------------------------
    Each instance is a member of a single, specific cluster.
    Each cluster is combined with a single, arbitrary label.
    Thus, each instance is assigned to a single, specific label,
    e.g., this creates 3 clusters, and links each instance to one of them:
    -- assign h 3

    Assign labels using fuzzy clustering (to create a multilabel dataset)
    --------------------------------------------------------------------------------------
    Each instance is a member of each cluster to some degree.
    This degree of membership is described by a non-negative real number in [0, 1].
    The sum of the degrees of membership of an instance sums up to 1.
    Each cluster is combined with a single, arbitrary label.
    Each instance is assigned to one label at least (the one with highest membership).
    The maxlabels and fuzzfactor settings determine which assignments are kept/pruned.

    e.g., this creates 4 clusters, and keeps only the two labels of highest membership
    per instance:
    -- set fuzzfactor 0.
    -- set maxlabels 2
    -- assign f 4

    e.g., this creates 4 clusters, and keeps labels with membership >= 0.7 v_{\mu}:
    -- set fuzzfactor 0.7
    -- set maxlabels None
    -- assign f 4

    Assign label rankings using fuzzy clustering (to create a label ranking dataset)
    --------------------------------------------------------------------------------------
    The same as before, except that the degree of membership is used to rank the labels
    e.g., this creates 4 clusters, and keeps all labels in the resulting rank:
    -- set fuzzfactor 0.
    -- set maxlabels None
    -- assign r 4
    """

    if(self.state < ECO_LOADED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()

      if(args[0] == 'o'):
        if(self.dataset.atype == ECO_DB_UNLABELLED):
          print(f'-- this dataset has no original labels')
          self.increase_exc()
          return False

        self.assignmethod = ECO_ASSIGNMENT_O
        procedure = (self.assignmethod, )
        ngroups = len(self.bk_target_names)
        self.dataset.atype        = self.bk_atype
        self.dataset.class_noun   = self.bk_class_noun
        self.dataset.target_names = self.bk_target_names
        self.showprotos = True

      elif(args[0] == 'h'):
        self.assignmethod = ECO_ASSIGNMENT_H
        procedure = (self.assignmethod, 'ward')
        ngroups = int(args[1])
        self.dataset.atype        = ECO_DB_MULTICLASS
        self.dataset.class_noun   = self.class_noun
        self.dataset.target_names = [f'{self.class_noun} {j}' for j in range(ngroups)]
        self.showprotos = True

      elif(args[0] == 'f'):
        self.assignmethod = ECO_ASSIGNMENT_F
        procedure = (self.assignmethod, )
        ngroups = int(args[1])
        self.dataset.atype        = ECO_DB_MULTILABEL
        self.dataset.class_noun   = self.class_noun
        self.dataset.target_names = [f'{self.class_noun} {j}' for j in range(ngroups)]
        self.showprotos = True

      elif(args[0] == 'r'):
        self.assignmethod = ECO_ASSIGNMENT_R
        procedure = (self.assignmethod, )
        ngroups = int(args[1])
        self.dataset.atype        = ECO_DB_LABELRANK
        self.dataset.class_noun   = self.class_noun
        self.dataset.target_names = [f'{self.class_noun} {j}' for j in range(ngroups)]
        self.showprotos = False

      self.assignpars = (procedure, False, None, self.maxlabels, self.random_state())
      (self.Y, self.U) = assign_groups(self.dataset,
                                      ngroups,
                                      params = self.assignpars,
                                      fuzzfactor=self.fuzzfactor)
      self.reset(ECO_ASSIGNED)


    except Exception as e:
      self.print_exc()
      self.reset(ECO_LOADED)
      return False

    self.dataset.assignments[ngroups] = self.Y # we save it here to make assignment2text easier
    self.dataset.memberships[ngroups] = self.U

  def do_split(self, line):
    """Assign instances (i.e., dataset rows) to either training or test partition, e.g.,
    Attempt to preserve the original proportion of labels in each partition
    -- split (uses the mean-vector splitting strategy with max_tries = 10000)
    -- split -s mvs -t 10000 (the same as 'split' without arguments)
    -- split -t 20000 (uses the mvs strategy with max_tries=20000)

    Attempt to force balance between labels in multiclass datasets
    In other words, class imbalance will be zero, at the cost of discarding samples
    -- split -fb (uses the mean-vector splitting strategy with force_balance=True)

    Attempt to ensure rare labelsets appear in splits (for multilabel/ranking datasets)
    -- split -s stv (uses the 'Sechidis, Tsoumakas, and Vlahavas, (2011)' strategy)
    """

    if(self.state < ECO_ASSIGNED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()
      if('-s' in args):
        pos = args.index('-s')
        strategy = args[pos+1].lower() # 'mvs' or 'stv' expected
        if(strategy not in [ECO_SPLIT_MVS, ECO_SPLIT_STV]):
          print(f'Splitting strategy {strategy} not implemented')
          self.increase_exc()
          return False

      else:
        strategy = ECO_SPLIT_MVS

      if('-t' in args):
        pos = args.index('-t')
        max_tries = int(args[pos+1])
      else:
        max_tries = 10000

      if('-fb' in args):
        force_balance=True
      else:
        force_balance=self.balance

      avoid_reset = ('-ar' in args)

      # splits the dataset into training and test partitions
      scenario = self._get_evaluation_scenario()
      (te_idxs, tr_idxs, success, last, tries) = split_dataset(
                                                   self.X, self.Y,
                                                   scenario      = scenario,
                                                   test_fraction = self.testfrac,
                                                   strategy      = strategy,
                                                   force_balance = force_balance,
                                                   threshold     = 0.01,
                                                   max_tries     = max_tries,
                                                   random_seed   = self.random_state(),
                                                 )

      self.te_idxs = te_idxs
      self.tr_idxs = tr_idxs
      if(not avoid_reset):
        self.reset(ECO_SPLIT)
      self.splitmethod = strategy
      self.splitpars   = args
      self.splitdesc   = strategy + self._desc_assignments(line='', verbose=False)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_ASSIGNED)
      return False

    if(not success):
      self.splitmsg = f'success: {success}, lowest watermark: {last} in {tries} attempts'
    else:
      self.splitmsg = None

  def do_train(self, line):
    """Trains a Polygrid instance with specified hyperparameters, e.g.,

    Trains a Polygrid model on the Iris dataset, with nspd = 1 and na = 2:
    -- load iris
    -- assign o
    -- split
    -- train 1 2
    -- ls

    The training of a Polygrid model is subject to many settings, such as:
    -- load iris
    -- assign o
    -- upgrade
    -- split
    -- set sector miss      (type of sector)
    -- set annulus tree     (type f annuli)
    -- set vorder averages  (method that determines how vertices are ordered)
    -- set solver ridge     (method used to solve a system of linear equations)
    -- set cutoffs multiple (toggles between single and multiple cutoffs)
    -- train 2 5
    -- ls
    """

    if(self.state < ECO_SPLIT):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()
      (nspd, na) = (int(args[0]), int(args[1]))

      # assesses the performance of selected model/config
      scenario = self._get_evaluation_scenario()
      self.model_type = 'Polygrid'
      self.model_config = build_Polygrid_config(nspd, na, self.polygrid)
      self.model = get_model(self.model_type, (self.model_config, self.random_state(), self.maxcores))
      self.model.set_dataset_name(self.dataset.name)
      self.model.set_target_names(self.dataset.target_names)
      self.model.set_feature_names(self.dataset.feature_names)
      self.model.set_nouns(self.dataset.sample_noun, self.dataset.class_noun)
      self.model.set_symbol(self.dataset.symbol)
      self.model.set_scenario(scenario)
      self.model.set_cutoff(self.cutoff)
      self.model.set_rcond(self.rcond)
      self.model.set_metric(self.metric)
      self.model.set_avoid_FP(self.avoid_FP)
      self.model.fit(self.X[self.tr_idxs], self.Y[self.tr_idxs], self.U[self.tr_idxs])
      self.model_size = self.model.get_sizing_data()
      # ensures the current setting for class order is observed
      self.model.setup_corder(self.corder)

      self.Y_real = self.Y[self.te_idxs]
      (self.Y_pred, self.YorU_hat) = self.model.predict(self.X[self.te_idxs], return_scores=True)
      (res, report) = evaluate(self.Y_real, self.Y_pred, self.YorU_hat, scenario)
      self.performance = (res, report)
      self.performstat = 'test'

      self.reset(ECO_TRAINED)
      self.last_nspd_na = (nspd, na)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_SPLIT)

  def do_fit(self, line):
    """Trains instances of competing models with size determined by the Polygrid instance trained last, e.g.,

    This trains a Polygrid model on the Iris dataset, with nspd = 1 and na = 2
    -- load iris
    -- assign o
    -- split
    -- train 1 2
    -- ls
    This creates a Polygrid model with 24 weighs.
    Then, if you issue:
    -- fit MLP
    -- ls
    An single-layered MLP model will be trained on exactly the same data, and
    the number of neurons will be determined so that, on the long running average,
    the model has about 24 weights.

    The list of competing models:
    -- fit MLP      (MLP w/ single hidden layer, logistic activivation, lbfgs, from sklearn)
    -- fit Linear   (Linear regressor, solver based on the LAPACK xgelsd drivers, no bias term is used, from sklearn)
    -- fit Ridge    (Ridge regressor,  solver based on the LAPACK xgelsd drivers, CV to adjust alpha, from sklearn)
    -- fit DT       (Decision Tree regressor, CART implementation from sklearn)
    -- fit BRDT     (Binary Relevance DT regressor, BR from scikit-multilearn-ng, DT from sklearn)
    -- fit RF       (Random Forest regressor, Breitman implementation from sklearn)
    -- fit BRRF     (Binary Relevance RF regressor, BR from scikit-multilearn-ng, DT from sklearn)
    -- fit Random   (Random regressor, custom, returns a random matrix in [0,1)^(m x n))

    About the LAPACK xgelsd drivers: https://www.netlib.org/lapack/lug/node27.html
    Binary Relevance is a commonly used problem-transformation technique in multilabel tasks.
    In DT, BRDT, and BRRF, the DT Regressor uses parameters from Fotakis et al., 2022.
    """
    #xxx review last paragraph, we want DT,RFDT to use params in Fotakis,
    #    but BRRF should follow Rivolli's parameters used in the foodtruck article

    if(self.state < ECO_SPLIT):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    if(self.model_size is None):
      print(f'-- you must first set the target model size, e.g., set size 100')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()

      # assesses the performance of selected model/configs
      scenario = self._get_evaluation_scenario()
      self.model_type = args[0]
      self.model_config = (self.model_size, self.random_state(), self.maxcores)
      self.model = get_model(self.model_type, self.model_config, self.model_hist)
      self.model.set_dataset_name(self.dataset.name)
      self.model.set_target_names(self.dataset.target_names)
      #self.model.set_feature_names(self.dataset.feature_names)
      #self.model.set_nouns(self.dataset.sample_noun, self.dataset.class_noun)
      #self.model.set_symbol(self.dataset.symbol)
      self.model.set_scenario(scenario)
      self.model.set_cutoff(self.cutoff)
      #self.model.set_rcond(self.rcond)
      self.model.set_metric(self.metric)
      self.model.set_avoid_FP(self.avoid_FP)
      self.model.fit(self.X[self.tr_idxs], self.Y[self.tr_idxs], self.U[self.tr_idxs])
      # self.model_size is not updated by competing models, but evalauate(..) saves it



      self.Y_real = self.Y[self.te_idxs]
      (self.Y_pred, self.YorU_hat) = self.model.predict(self.X[self.te_idxs], return_scores=True)
      (res, report) = evaluate(self.Y_real, self.Y_pred, self.YorU_hat, scenario)
      self.performance = (res, report)
      self.performstat = 'test'

      self.reset(ECO_TRAINED)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_SPLIT)

  def do_setup(self, line):
    """Trains with a Polygrid instance with parameters from an offline evaluation, e.g.,

    >> python cli.py ..\configs\evaluate_H01_C1.cfg
    -- load iris
    -- setup 64

    This will train a Polygrid instance with the config with index 64:
    (1, 2, 'averages', 'r-invariant', 'miss', 'lstsqsym', 'single')

    The config spec is recovered from the results of the offline evaluation that is referred
    in the file PARAM_FILENAME of the configuration file used to start the cli environment
    """
    if(self.state < ECO_LOADED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    if(self.offlineval_configs is None):
      targetpath = self.targetpath
      filename   = self.offlineval
      print(f'-- loading results from {join(*targetpath, filename)}.pkl')
      (datasets, datapars, best_polygrid_config, competitors, polygrid_configs, all_results) = deserialise(join(*targetpath, filename))
      self.offlineval_pending_upd = True
      self.offlineval_datasets    = datasets
      self.offlineval_datapars    = datapars
      self.offlineval_competitors = competitors
      self.offlineval_results     = all_results
      self.offlineval_configs     = polygrid_configs

    try:
      args = line.strip().split()
      k = int(args[0])
      (nspd, na, vorder, annulus, sector, solver, cutoff) = self.offlineval_configs[k]

      # recovers and applies the dataset-related parameters of the currently loaded
      # dataset from the configuration file used to start the cli environment
      found = False
      for (i, dataset_lbl) in enumerate(self.offlineval_datasets):
        (dataset, fuzzfactor, assignpars, splitpars) = self.offlineval_datapars[dataset_lbl]
        if(dataset == self.dataset.name):

          self.do_set(f' fuzzfactor {fuzzfactor}')
          self.do_assign(f' {assignpars}')
          self.do_split(f' {splitpars}')

          print(f'-- assign {assignpars}')
          print(f'-- split {splitpars}')

          found = True
          break

      # recovers and applies the model-related parameters and trains a Polygrid model
      if(found):
        self.do_set(f' vorder {vorder}')
        self.do_set(f' annulus {annulus}')
        self.do_set(f' sector {sector}')
        self.do_set(f' solver {solver}')
        self.do_set(f' cutoff {cutoff}')
        self.do_train(f' {nspd} {na}')

        j = self.offlineval_competitors.index('Polygrid')
        print(f'-- recovered from entry {(i,j,k)}')
      else:
        raise ValueError(f'Dataset {self.dataset.name} not found in the results of the offline evaluation')

    except:
      self.print_exc()
      self.reset(ECO_LOADED)

  def do_eval(self, line):
    """Performs a comparative evaluation of the relevant models, e.g.,

    Using the configuration file for general multiclass datasets:
    >> python cli.py ..\configs\evaluate_H01_C1.cfg
    -- set runs 5
    -- load cancer
    -- eval 64

    This will train a Polygrid instance with the config with index 64:
    (1, 2, 'averages', 'r-invariant', 'miss', 'lstsqsym', 'single')
    And also fit the alternative models constrained to the size of the Polygrid model

    The config spec is recovered from the results of the offline evaluation that is referred
    in the file PARAM_FILENAME of the configuration file used to start the cli environment
    """

    if(self.state < ECO_LOADED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    if(self.offlineval_configs is None):
      targetpath = self.targetpath
      filename   = self.offlineval
      print(f'-- loading results from {join(*targetpath, filename)}.pkl')
      (datasets, datapars, best_polygrid_config, competitors, polygrid_configs, all_results) = deserialise(join(*targetpath, filename))
      self.offlineval_pending_upd = True
      self.offlineval_datasets    = datasets
      self.offlineval_datapars    = datapars
      self.offlineval_competitors = competitors
      self.offlineval_results     = all_results
      self.offlineval_configs     = polygrid_configs

    try:
      args = line.strip().split()
      k = int(args[0])
      (nspd, na, vorder, annulus, sector, solver, cutoff) = self.offlineval_configs[k]

      # recovers and applies the dataset-related parameters of the currently loaded
      # dataset from the configuration file used to start the cli environment
      found = False
      for dataset_lbl in self.offlineval_datasets:
        (dataset, fuzzfactor, assignpars, splitpars) = self.offlineval_datapars[dataset_lbl]
        if(dataset == self.dataset.name):

          self.do_set(f' fuzzfactor {fuzzfactor}')
          self.do_assign(f' {assignpars}')
          self.do_split(f' {splitpars}')

          print(f'-- assign {assignpars}')
          print(f'-- split {splitpars}')

          i = self.offlineval_datasets.index(dataset_lbl)
          found = True
          break

      # recovers and applies the model-related parameters and evaluates the models
      if(found):

        # recovering Polygrid performance data from the offline evaluation
        print('-- recovering sizing data for Polygrid')
        j0 = self.offlineval_competitors.index('Polygrid')
        self.model_size = self.offlineval_results[i,j0,k]['sizing']

        print('-- assessing performance of the alternative models')
        for (j, model_type) in enumerate(self.offlineval_competitors):
          if(model_type != 'Polygrid'):
            print(f'-- fitting {model_type}')
            self.do_fit(f'{model_type}')
            self.do_assess('')
            self.offlineval_results[i,j,k] = self.performance[0]
            print(self.performance[1])
            print(f'   results updated at {(i,j,k)}')

      else:
        raise ValueError(f'Dataset {self.dataset.name} not found in the results of the offline evaluation')

    except:
      self.print_exc()
      self.reset(ECO_LOADED)

  def do_retrain(self, line):
    """Trains a Polygrid instance using currently specified hyperparameters, e.g.,

    Trains a Polygrid model on the Iris dataset, with nspd = 1 and na = 2
    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- ls
    -- set sector miss
    -- retrain
    """

    if(self.state < ECO_SPLIT or self.last_nspd_na is None):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      (nspd, na) = self.last_nspd_na
      self.do_train(f'{nspd} {na}')

    except:
      self.print_exc()
      self.reset(ECO_SPLIT)

  def do_assess(self, line):
    """Repeats the split-train or fit-evaluate cycle a number of times to compute the confidence 
    intervals of relevant performance statistics, e.g., 
    -- set runs 50
    -- load iris
    -- assign o
    -- split -t 100
    -- train 1 2
    -- assess
    This will repeat {split -t 100; train 1 2} for 50 times and compute the
    confidence interval for the relevant performance statistics.

    -- fit MLP
    -- assess
    -- ls
    This will repeat {split -t 100; fit MLP} for 50 times and so on.
    """
    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      start_time = time.time()
      (evaluation, report) = self.performance
      if('attc' in evaluation):
        # last evaluation was produced by do_assess; start anew
        evaluations = []
        sizes = []
        cutoffs = []
        offset = 0
      else:
        # last evaluation was produced by do_train or do_fit; reuse current evaluation
        evaluations = [evaluation]
        sizes = [self.model.get_size()]
        if(self.model.tw is not None):
          cutoffs = self.model.tw.tolist()
        else:
          cutoffs = []
        offset = 1
      model_type = self.model_type
      if(model_type == 'Polygrid'):
        (nspd, na) = (self.model.nspd, self.model.na)
      for run in range(self.essayruns - offset):
        if(self.verbose):
          print(f'-- run {run + 1 + offset}/{self.essayruns}')
        self.cmdcounter += 1
        self.do_split(' '.join(self.splitpars + ['-ar']))
        if(model_type == 'Polygrid'):
          self.do_train(f'{nspd} {na}')
        else:
          self.model_hist = sizes # will be passed on to competitor init
          self.do_fit(model_type)
        (evaluation, report) = self.performance
        evaluations.append(evaluation)
        sizes.append(self.model.get_size())
        if(self.model.tw is not None):
          cutoffs += self.model.tw.tolist()

      scenario = self._get_evaluation_scenario()

      (res, report) = estimate(evaluations, scenario, self.alpha)
      res['sizing'] = self.model_size
      res['sizes'] = [float(size) for size in sizes] # for some reason, list of integers
                                                     # cannot be serialised in
                                                     # celery/kombu (raises an error),
                                                     # but no issue with list of floats!
      res['attc'] = (time.time() - start_time)/(self.essayruns - 1)
      res['cutoffs'] = cutoffs
      report.add(f'   av. size: {np.mean(sizes):5.1f}')
      self.performance = (res, report)
      self.performstat = 'test'
      self.model_hist = []

    except:
      self.print_exc()

  #-------------------------------------------------------------------------------------
  # specifies data/model inspection commands
  #-------------------------------------------------------------------------------------

  def do_config(self, line):
    """Describes the context and the configuration of the instantiated model.
    """
    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    catalog_entry = None
    if(self.offlineval_configs is not None):
      if(self.model.model_name == 'Polygrid'):
        spec = (self.model.nspd,
                self.model.na,
                self.model.vorder,
                self.model.annulus_type,
                self.model.sector_type,
                self.model.solver,
                self.model.cutoff)
        try:
          catalog_entry = f'setup: {self.offlineval_configs.index(spec)}'
        except ValueError:
          pass

    try:
      print('-- Summary of current context and model configuration')
      config = self.model.get_config_summary(include_model_name=True, details=[])
      if(catalog_entry is not None):
        config.append(catalog_entry)
      (params, vals) = list(zip(*[pair.split(':') for pair in config]))
      maxlen = max([len(param) for param in params])
      mask = '   {0:' + str(maxlen) + 's} {1}'
      for (param, val) in zip(params, vals):
        print(mask.format(param, val))

    except Exception as e:
      self.print_exc()

  def do_tree(self, line):
    """Generates a graphviz script with the content of a tree, e.g.,

    Display the structure of a decision tree model (DT):
    -- load iris
    -- assign o
    -- split
    -- set size free
    -- fit DT
    -- tree

    For multi-tree models, the specific tree to be described must be identified:
    -- set size narrow
    -- fit RF
    -- tree 0
    -- tree 1
    -- tree 2

    The generated script can be graphically rendered by using online services such as:
    https://dreampuf.github.io/GraphvizOnline/
    """

    def shorten(s):
      L = s.replace('-', ' ').split(' ')
      if(len(L) > 1):
        res = str(L[0][0] + L[1][0]).lower()
      else:
        res = L[0][0:2].lower()
      return res

    if(self.model is None or self.model.model_name not in ['DT', 'RF', 'BRDT', 'BRRF']):
      print(f'-- this command is only enabled for tree-based models')
      return False

    args = line.strip().split()
    j = int(args[0]) if len(args) > 0 else None

    try:

      # recovers string translations for the current dataset
      (column2label, _, _) = get_labels(self.dataset)
      feature_names = [column2label[feature_name] for feature_name in self.dataset.feature_names]

      # determines which tree will be detailed
      if(self.model.model_name == 'DT'):
        tree = self.model.model
      elif(self.model.model_name == 'BRDT'):
        tree = self.model.model.classifiers_[j]
      elif(self.model.model_name == 'RF'):
        tree = self.model.model.estimators_[j]
      elif(self.model.model_name == 'BRRF'):
        tree = self.model.model.classifiers_[j].estimators_[0]

      # prints a .dot script of the required tree
      diagram = export_graphviz(tree,
                      feature_names=feature_names,
                      impurity=False,
                      filled=True,
                      precision=2)

      diagram = diagram.replace('shape=box', 'shape=circle')
      diagram = diagram.replace('True',  '<=')
      diagram = diagram.replace('False', '>')
      diagram = diagram[:-2] + 'ranksep=0.5; nodesep=1.5;\n}'

      for feature_name in self.dataset.feature_names:
        diagram = diagram.replace(feature_name, shorten(feature_name))

      diagram = re.sub(r'label="(.*) <= (\d\.\d+).*value = .*"', r'label="\1", xlabel="\2", fillcolor="#ffffff"', diagram)
      diagram = re.sub(r'label="samples = \d+\\nvalue = (.*)"', r'label="\1"', diagram)

      scenario = self._get_evaluation_scenario()
      if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        diagram = re.sub(r'(\[)*0.0(\])*(\\n)*', r'0', diagram)
        diagram = re.sub(r'(\[)*1.0(\])*(\\n)*', r'1', diagram)
      #  (_, n) = self.Y.shape
      #  v = np.zeros(n, dtype=int)
      #  for j in range(n):
      #    ind = str(j)
      #    v[:] = 0
      #    v[j] = 1
      #    s = ''.join([str(e) for e in v.tolist()])
      #    #print(s)
      #    diagram = re.sub(r'label="{0}"'.format(s), r'label="{0}"'.format(ind), diagram)

      elif(scenario in [ECO_DB_LABELRANK]):

        def compnode(match):
          res = match.group()
          try:
            aux = res.replace('label=', '')
            aux = aux.replace('"', '')
            aux = aux.replace('\\n', ', ')
            aux = eval(aux)
            aux = (-np.array(aux)).T[0].argsort()
            aux = 'label="[' + ''.join([str(e) for e in aux.tolist()]) + ']"'
            return aux
          except:
            return res

        diagram = re.sub(r'label="\[(.*)\]"', compnode, diagram)


      else:
        raise ValueError('Support for {scenario} task not implemented.')

      print(diagram)

    except Exception as e:
      self.print_exc()

  def do_inspect(self, line):
    """Inspects the performance of a trained/fitted model, similar to 'show scales' does, e.g.,

    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- inspect class 0    (inspects Polygrid's performance on instances of class 0)
    -- fit MLP
    -- inspect class 0    (inspects MLP's performance on instances of class 0)
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      args = line.strip().split()
      (subcmd, *args) = args

      if(subcmd == 'class'):
        j = int(args[0])
        anchor = self.model.P[j][self.model.get_reverse_vorder()]
        anchor_lbl = f'prototype of class {j}'

      elif(subcmd == 'classof'):
        i = int(args[0])
        j = self.dataset.target[i]
        anchor = self.dataset.data[i]
        anchor_lbl = f'instance {i}'

      else:
        print(f'Sub-command not recognised: inspect {subcmd}')
        return False

      (m,n) = self.Y.shape
      all_idxs = list(range(m))
      if(self.inspection is None):
        self.inspection = self.model.inspect(self.X, self.Y, all_idxs, self.tr_idxs)
      (Y_pred, YorU_hat, names, coords, cutoffs, yticklabels) = self.inspection
      y2pc = {-2: ('te', 'TP'), -1: ('te', 'TN'), 1: ('tr', 'TN'), 2: ('tr', 'TP'), }

      partitions = ['tr', 'te']
      contingencies = ['TN', 'TP']
      outcomes = [ECO_HIT, ECO_MISS]

      report = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
      for outcome in outcomes:
        for (i, idx) in enumerate(names[j][outcome]):
          (x,y) = coords[j][outcome][i]
          (partition, contingency) = y2pc[y]
          score = x
          report[partition][contingency][outcome].append((idx, score))

      # prints the report
      print(f'anchor type: {anchor_lbl}')
      for partition in partitions:
        print()
        for contingency in contingencies:
          print(f'-- {partition}/{contingency}:')
          for outcome in outcomes:
            samples = report[partition][contingency][outcome]
            ss = len(samples)
            buffer = rank_key_by_value(samples, clip_value=False, decimals=4)
            print(f'   {ss:3d} {outcome}: {buffer[0:self.topN]}')

    except Exception as e:
      self.print_exc()

  def do_diag(self, line):
    """Shows a report analysing the classes of errors made by the trained instance, e.g.,

    Describes the types of errors in a multiclass task (only FP or FN)
    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- diag

    Describes the types of errors in a multilabel task  (only FP or FN)
    -- upgrade
    -- retrain
    -- diag

    Describes the types of errors in a label ranking task  (FP, FN, inversions, bubbles)
    -- set fuzzfactor 0.1
    -- assign r 5
    -- split
    -- train 1 1
    -- diag
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      (m,n) = self.Y.shape
      scenario = self._get_evaluation_scenario()
      (report, header, Y_errstats) = diagnose(self.Y_real, self.Y_pred, self.YorU_hat, scenario)
      print(f'\n{report}')

      if(self.filename is not None):
        content = get_tikz_diag(self.Y_real, self.Y_pred, header, Y_errstats, scenario)
        saveAsText(content, join(*self.targetpath, self.filename))
        print(f'-- report has been saved to {join(*self.targetpath, self.filename)}')

    except Exception as e:
      self.print_exc()

  def do_desc(self, line):
    """Describes an instance, e.g.,

    Describes the instance in the 50th dataset row:
    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- desc 50

    Describes multiple instances
    -- desc 0 50 100
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      (m,n) = self.Y.shape
      feature_names = self.dataset.feature_names
      target_names  = self.dataset.target_names
      scenario = self._get_evaluation_scenario()
      args = line.strip().split()
      for idx in args:
        idx = int(idx)
        print('\nSample {0}, Subject {1}'.format(idx, self.dataset.caseIDs[idx]))
        print('Original scores ..: {0}'.format(self.dataset.odata[idx][self.model.vo]))
        print('Transformed scores: {0}'.format(self.dataset.data[idx][self.model.vo]))
        print('Attribute names: {0}'.format(feature_names))
        print('-- sum-score  .: {0}'.format(self.dataset.odata[idx].sum().round(decimals=1)))
        print('-- area-score .: {0}'.format(self.model.scores2poly(self.dataset.data[idx][self.model.vo]).area))

        if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
          assigned_classes = ['({0}) {1}'.format(j, target_names[j]) for j in range(n) if self.Y[idx][j] == 1]
          content = ', '.join(assigned_classes)
        elif(scenario in [ECO_DB_LABELRANK]):
          #assigned_classes = ['({0}) {1}'.format(j, target_names[j]) for j in range(n) if self.Y[caseID][j] >= 0]
          assigned_classes = ['({0}) {1}'.format(j, target_names[j]) for j in self.Y[idx] if j >= 0]
          content = ' > '.join(assigned_classes)
        print('Assigned class(es): {0}'.format(content))

      print()

    except Exception as e:
      self.print_exc()

  def do_whois(self, line):
    """Finds the ID (row number) of a case based on its caseID, e.g.

    Finds the position of the informed instance in the training or test partition:
    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- desc 50
       Sample 50, Subject P50
       Original scores ..: [7.  3.2 4.7 1.4]
       Transformed scores: [0.88607595 0.72727273 0.68115942 0.56      ]
       Attribute names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
       -- sum-score  .: 16.3
       -- area-score .: 1.0087296742882872
       Assigned class(es): (1) versicolor
    -- whois P50
       -- Case ID P50 is at position #50 of the dataset
          Case ID P50 is at position #12 of the training partition
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      args = line.strip().split()
      caseID = args[0]
      dsid   = self.dataset.caseIDs.index(caseID) # position of the case in dataset (row number)
      try:
        part = 'training'
        ptid = self.tr_idxs.tolist().index(dsid)  # position of the case in the training partition
      except:
        part = 'test'
        ptid = self.te_idxs.tolist().index(dsid)  # position of the case in the test partition

      print('-- Case ID {0} is at position #{1} of the dataset'.format(caseID, dsid))
      print('   Case ID {0} is at position #{1} of the {2} partition'.format(caseID, ptid, part))

    except Exception as e:
      self.print_exc()

  def do_list(self, line):
    """List the first N instances allocated to the train and test partitions, e.g.,
    -- list 5
       -- train: [ 35 141  64  51  99]
          test : [112 135  82 108  40]
    """

    if(self.state < ECO_SPLIT):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      args = line.strip().split()
      firstN = int(args[0]) if len(args) > 0 else self.topN
      print('-- train: {0}'.format(self.tr_idxs[0:firstN]))
      print('   test : {0}'.format(self.te_idxs[0:firstN]))

    except Exception as e:
      self.print_exc()

  def do_range(self, line):
    """Evaluates the range of measures (i.e., areas) of instances in the train and test partitions, e.g.,

    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- range
       -- train: measures range from 0.28080302196427676 to 1.6962949250346058
          test : measures range from 0.24360200797184828 to 1.4917836593785958

    In general, the ranges for both partitions should be similar.
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      self. measures = {}
      (m, d) = self.X.shape
      for i in range(m):
        self.measures[i] = self.model.coords2poly(self.model.scores2coords(self.X[i,self.model.vo])).area

      for (label, idxs) in [('-- train:', self.tr_idxs), ('   test :', self.te_idxs)]:
        L = [self.measures[i] for i in idxs]
        lb = min(L)
        ub = max(L)
        print('{0} measures range from {1} to {2}'.format(label, lb, ub))

    except Exception as e:
      self.print_exc()
      self.measures = None

  def do_search(self, line):
    """Based on extremes shown by command range, searches instances whose measure falls within a given interval, e.g.,

    Displays up to 4 instances whose measure in the [0.5, 0.8] interval):
    -- load iris
    -- assign o
    -- split
    -- train 1 1
    -- range
       -- train: measures range from 0.24360200797184828 to 1.6962949250346058
          test : measures range from 0.2539804206067277 to 1.4917836593785958
    -- search  .5 .8 4
       -- train: [(93, 0.5126544754090158), (57, 0.519297543403212), (15, 0.5445679691799669), (98, 0.544595737229199)]
          test : [(62, 0.6026417171161255), (94, 0.7468149297043077), (68, 0.7903412217941662), (106, 0.7941101716115474)]

    If the number of instances is no informed, then topN items are shown:
    -- set topN 2
    -- search  .5 .8
       -- train: [(93, 0.5126544754090158), (57, 0.519297543403212)]
          test : [(62, 0.6026417171161255), (94, 0.7468149297043077)]
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      args = line.strip().split()
      lb = float(args[0])
      ub = float(args[1])
      try:
        topN = int(args[2])
      except:
        topN = self.topN

      if(self.measures is None):
        self.measures = {}
        (m, d) = self.X.shape
        for i in range(m):
          self.measures[i] = self.model.coords2poly(self.model.scores2coords(self.X[i,self.model.vo])).area

      for (label, idxs) in [('-- train:', self.tr_idxs), ('   test :', self.te_idxs)]:
        L = sorted([(i, self.measures[i]) for i in idxs if lb <= self.measures[i] <= ub], key=lambda e:e[1])
        print('{0} {1}'.format(label, L[0:topN]))

    except Exception as e:
      self.print_exc()

  def do_nearest(self, line):
    """Finds the k-nearest neighbours of an instance or prototype, e.g.,

    -- nearest 3 from 84  (finds the 3 nearest neighbours of instance 84)
    -- nearest 5 of 0     (finds the 5 nearest neighbours of class prototype 0)

    The reciprocal of the euclidian distance in the source space is used as 'nearness'
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    try:
      args = line.strip().split()
      (k_neighbours, preposition, anchor) = args
      k_neighbours = int(k_neighbours)
      anchor = int(anchor)

      if(preposition == 'from'):
        anchor_type = 'instance'
        anchor_v    = self.X[anchor]

      elif(preposition == 'of'):
        anchor_type = 'class'
        anchor_v    = self.model.P[anchor][self.model.get_reverse_vorder()]

      else:
        print(f'-- wrong syntax: preposition {preposition} should be wither from or of.')
        return False

      tr_neighbours = [(idx, np.linalg.norm(anchor_v - self.X[idx])) for idx in self.tr_idxs]
      te_neighbours = [(idx, np.linalg.norm(anchor_v - self.X[idx])) for idx in self.te_idxs]

      tr_neighbours = sorted(tr_neighbours, key = lambda e: e[1])
      te_neighbours = sorted(te_neighbours, key = lambda e: e[1])

      print(f'These are the {k_neighbours} nearest neighbours of {anchor_type} {anchor}:')

      if(self._get_evaluation_scenario() in [ECO_DB_MULTICLASS]):
        print('-- training:')
        for (idx, dist) in tr_neighbours[0:k_neighbours]:
          print(f'   case {idx}: distance {dist:10.8f} (of class {self.Y[idx].argmax()}, classified as {self.model.predict(self.X[idx,:].reshape((1, self.model.d))).argmax()})')
        print('-- test:')
        for (idx, dist) in te_neighbours[0:k_neighbours]:
          print(f'   case {idx}: distance {dist:10.8f} (of class {self.Y[idx].argmax()}, classified as {self.model.predict(self.X[idx,:].reshape((1, self.model.d))).argmax()})')

      else:
        print('-- training:')
        for (idx, dist) in tr_neighbours[0:k_neighbours]:
          print(f'   case {idx}: distance {dist:10.8f}')
        print('-- test:')
        for (idx, dist) in te_neighbours[0:k_neighbours]:
          print(f'   case {idx}: distance {dist:10.8f}')

    except:
      self.print_exc()


  #-------------------------------------------------------------------------------------
  # specifies the show commands
  #-------------------------------------------------------------------------------------

  def do_show(self, line):
    """Display diagrams for distinct entities in the Polygrid model, e.g.,

    1. Demands and offers:
    -- show demand 84      # displays diagram with demand profile
    -- show demands 66 84  # displays diagram with multiple demand profiles
    -- show offers         # displays diagram with offer profiles
    -- show cross 84       # displays a Polygrid diagram (demands and offers)
    -- show cross 66 84
    -- show bars 84        # displays a Barsgrid diagram (demands and offers)
    -- show bars 66 84
    -- show scales         # displays the Polygrid scales for the whole dataset
    -- show scale 0        # displays the Polygrid scales for label 0
    -- show scales [0,1,2] # displays the Polygrid scales for labelset (0,1,2)
    -- show scale -4       # displays the Polygrid scales for the whole dataset and
                           #   selects/highlights instance 4

    2. Distributions of instance around clusters
    -- show clusters       # displays training and test instances overlaid per label
                            (training instances in solid lines, test in dashed lines)
    -- show clusters 10 20 # displays training instances + instances 10 and 20 (dashed)

    3. Distributions of mass over cells (not quite working yet)
    -- show mass           # displays the distribution of mass per cell
    -- show prob           # displays the probability per cell

    4. Plot diagrams illustrating different aspects of the representation space
    -- show disc           # displays the partitioned unit disc, no labels
    -- show omega          # displays the partitioned unit disc, cell labels only
    -- show frames         # displays the partitioned unit disc, domain labels only
    -- show space          # displays the partitioned unit disc, with cell and domain labels

    5. Plot user-defined objects (of the same dimension of the loaded dataset)
    -- show plot [[.5, .5, .5, .5], ]
    -- show plots [[.5, .5, .5, .5], [.5, .6, .5, .6], ]
    -- show piece [[.4, .4, 0., 0.], ]
    -- show pieces [[.4, .4, 0., 0.], [0., 0., .6, .6], [.5, .5, .5, .5], ]
    """

    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      return False

    #if(self.model_type not in ['Polygrid']):
    #  print(f'-- command is not enabled for the current model: {self.model_type}')
    #  return False

    # create aliases to PolygridCLI attributes related to ...

    # ... parameters and ...
    gridCols   = self.gridCols
    scaleCols  = self.scaleCols
    transpose  = self.transpose
    hide_tags  = self.hide_tags
    filename   = self.filename
    usecaseids = self.usecaseids

    # ... objects
    dataset = self.dataset
    X       = self.X
    Y       = self.Y
    tr_idxs = self.tr_idxs
    te_idxs = self.te_idxs
    model   = self.model
    Y_real  = self.Y_real
    Y_pred  = self.Y_pred

    # an instance is assigned to the cluster which it attains the highest membership
    # for multilabel and label ranking, an instance may be assigned to multiple clusters
    #_Y = (self.U == self.U.max(axis=1)[:, None]).astype(int)
    scenario = self._get_evaluation_scenario()
    if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
      _Y = self.Y
    elif(scenario == ECO_DB_LABELRANK):
      _Y = rank2presence(self.Y)
    else:
      raise ValueError(f'Scenario {scenario} is not handled by the show commands')
    label_counts = _Y.sum(axis=0)

    try:
      (column2label, offset, alignment) = get_labels(dataset)
      if(self.column2label is not None):
        column2label = self.column2label
        self.model.set_dataset_name(column2label[self.dataset.name])
      layout_data['dimension.label.offset'] = offset
      layout_data['dimension.label.alignment'] = alignment
      show_params = (gridCols, transpose, column2label, layout_data, hide_tags, filename)

      args = line.strip().split()
      (subcmd, *args) = args

      if(self.model.model_name != 'Polygrid' and subcmd not in ['scales']):
        print(f'-- this command is only enabled for the Polygrid model')
        return False

      if(subcmd in ['demand', 'demands']):
        idxs = [int(arg) for arg in args] if len(args) > 0 else te_idxs[0:gridCols]
        perf_params = None
        show_idxs = [dataset.caseIDs[i] for i in idxs] if usecaseids else idxs
        model.show_demands(X[idxs,:], idxs, perf_params, show_params, show_idxs)
        print('-- shown')

      elif(subcmd in ['offers']):
        perf_params = label_counts
        model.show_offers(perf_params, show_params, show_protos=self.showprotos)
        print('-- shown')

      elif(subcmd in ['cross']):
        idxs = [int(arg) for arg in args] if len(args) > 0 else te_idxs[0:gridCols//2]
        perf_params = None
        show_idxs = [dataset.caseIDs[i] for i in idxs] if usecaseids else idxs
        show_protos = self.showprotos
        model.show_cross(X[idxs,:], idxs, perf_params, show_params, show_idxs, show_protos)
        print('-- shown')

      elif(subcmd in ['bars']):
        if(self.model.na > 1):
          print('Barsgrid diagram is not available for na > 1')
          return False
        idxs = [int(arg) for arg in args] if len(args) > 0 else te_idxs[0:gridCols]
        perf_params = None
        show_idxs = [dataset.caseIDs[i] for i in idxs] if usecaseids else idxs
        model.show_bar_cross(X[idxs,:], idxs, perf_params, show_params, show_idxs, scaling=dataset.scaling)
        print('-- shown')

      elif(subcmd in ['scale', 'scales']):
        (m,d) = X.shape
        (_,n) = Y.shape
        if(len(args) == 0):
          label = None
          labelset = None
          idxs = list(range(m))
          popup = None
        elif(isinstance(eval(args[0]), int)):
          j = eval(args[0])
          label = j
          labelset = None
          #if(0 <= j < n):
          if('-' in ''.join(args)):
            # parameter refers to a case id to be "highlighted" in the diagram
            idxs = list(range(m))
            popup = -j
          else:
            # parameter referes to the index of a class, so diagram will show
            # only positive instances of that class
            idxs = np.where(Y[:,j] == 1)[0]
            popup = None
        elif(isinstance(eval(''.join(args)), list)):
          L = [str(e) for e in Y.tolist()]
          label = None
          labelset = str(eval(args[0])) #str(eval(''.join(args[0])))
          idxs = np.where([e == labelset for e in L])[0]
          popup = None
        else:
          raise ValueError(f'Parameter {args} is invalid for show scales command.')

        if(len(idxs) > 0):
          perf_params = (scenario, popup,)
          model.show_scales(X[idxs,:], Y[idxs,:], idxs, tr_idxs, perf_params, show_params, scaleCols)
          print('-- shown')
        else:
          print('Label or labelset did not not identify any rows.')

      elif(subcmd in ['clusters']):
        if(len(args) > 0):
          idxs1 = tr_idxs
          idxs2 = [int(arg) for arg in args]
        else:
          idxs1 = tr_idxs
          idxs2 = te_idxs
        model.show_clusters( X,
                           _Y,
                           perf_params=(label_counts, idxs1, idxs2),
                           show_params=show_params)
        print('-- shown')

      elif(subcmd in ['mass', 'prob']):
        cmap_old = layout_data['cbar.cmap']
        layout_data['cbar.cmap'] = 'uniform'
        perf_params = (label_counts, subcmd)
        model.show_mass(Y, tr_idxs, perf_params, show_params)
        print('-- shown')
        layout_data['cbar.cmap'] = cmap_old

      elif(subcmd in ['plot', 'plots']):
        scores = np.array(eval(' '.join(args)))
        (_m,_) = scores.shape
        model.show_demands(scores, idxs=range(_m),
                          perf_params=None,
                          show_params=show_params)
        print('-- shown')

      elif(subcmd in ['piece', 'pieces']):
        scores = np.array(eval(' '.join(args)))
        (_m,_) = scores.shape
        model.show_clusters(scores,
                           np.ones((_m,1)),
                           perf_params=None,
                           show_params=show_params)
        print('-- shown')

      elif(subcmd == 'disc'):
        cmap_old = layout_data['cbar.cmap']
        layout_data['cbar.cmap'] = 'uniform'
        model.show_disc(show_params, domain_labels=False, cell_labels=False)
        layout_data['cbar.cmap'] = cmap_old
        print('-- shown')

      elif(subcmd == 'omega'):
        cmap_old = layout_data['cbar.cmap']
        layout_data['cbar.cmap'] = 'uniform'
        model.show_disc(show_params, domain_labels=False, cell_labels=True)
        layout_data['cbar.cmap'] = cmap_old
        print('-- shown')

      elif(subcmd == 'frames'):
        cmap_old = layout_data['cbar.cmap']
        layout_data['cbar.cmap'] = 'uniform'
        model.show_disc(show_params, domain_labels=True,  cell_labels=False)
        layout_data['cbar.cmap'] = cmap_old
        print('-- shown')

      elif(subcmd == 'space'):
        cmap_old = layout_data['cbar.cmap']
        layout_data['cbar.cmap'] = 'uniform'
        model.show_disc(show_params, domain_labels=True,  cell_labels=True)
        layout_data['cbar.cmap'] = cmap_old
        print('-- shown')

      else:
        print('-- unknown sub-command: {0}'.format(subcmd))

    except Exception as e:
      self.print_exc()




  #-------------------------------------------------------------------------------------
  # specifies state setting commands
  #-------------------------------------------------------------------------------------

  def do_set(self, line):
    """Sets environment variables to given value, e.g.,

    Settings for the Polygrid model
    --------------------------------------------------------------------------------------
    -- set annulus [s-invariant|r-invariant|tree|random]
    -- set corder [original|measures]
    -- set noun <string>
       Selects the noun that describes instances, e.g., 'Specimen' in the Iris dataset
       This noun is used in demand and cross diagrams
    -- set rcond <float|integer>
       A value in [0., 1) sets (value * first singular value) as rcond for lstsq
       An integer value 1 .. g indicates the number of singular vectors to include in lstsq
       The value -1 sets rcond as the machine precision
    -- set sector [cover|miss|random]
    -- set solver [lstsq|lstsqsym|ridge|lasso|braids]
    -- set vorder [original|averages|rho|rho-squared|measures]

    Settings for all models
    --------------------------------------------------------------------------------------
    -- set cutoff [single|multiple]
       For cutoff=single, all scales will share a single cutoff
       For cutoff=multiple, each scale will be allowed to have its own cutoff
    -- set size [free|narrow|<integer>]
       For MLP:  free = narrow, both allow for an MLP with 100 hidden neurons
       For DT:   free = narrow, both allow for 1 DT  of unconstrained depth
       For BRDT: free = narrow, both allow for n DTs of unconstrained depth
       For RF:   free   allows for 1 forest with 100 DTs of unconstrained depth
                 narrow allows for 1 forest with  n  DTs of unconstrained depth
       For BRRF: free   allows for n forests, each with 100 DTs of unconstrained depth
                 narrow allows for n forests, each with  n  DTs of unconstrained depth

    Setting for evaluation
    --------------------------------------------------------------------------------------
    -- set alpha 0.5    (sets the confidence level to be used in performance reports)
    -- set runs 30      (every model assessment will perform this number of repeats)
    -- set testfrac 0.2 (will use this fraction of the data as test data)

    Settings for visualisations
    --------------------------------------------------------------------------------------
    -- set cmap [diverging|uniform]
       Selects the colour map of demand/offer/cross diagrams
    -- set gridcols 3   (sets the number of columns in Polygrid diagrams)
    -- set norm [TwoSlopeNorm|SymLogNorm]
       Selects the norm used on colour maps for demand/offer/cross diagrams
    -- set scalecols 2  (sets the number of columns in scale diagrams)
    -- set topn 5       (sets the number of instances shown in command responses)

    Settings for CLI environment
    --------------------------------------------------------------------------------------
    -- set filename plot.png
       Sets the name of the output file. Behaviour vary depending on the file type
       For .tex and .pgf files, a 'tex' string is added to the target path
       For .jpg, .png, and .pdf files, a 'figures' string is added to the target path
       For other file types, the file will be saved in the target path directly
    -- set fuzzfactor <float>
       Sets the minimum degree of membership of an instance to a cluster.
       This is used in assign command for multilabel and label ranking assignments.
       In these assignments, fuzzy clustering is applied to the description matrix to
       produce synthetic assignments. This setting indicates the minimum degree of
       membership an instance must have to be assigned to the label associated with its
       corresponding cluster. So, the value must be in [0, 1].
    -- set maxlabels <integer>
       Sets the maximum number of labels assigned to an instance.
       This is used in assign command for multilabel and label ranking assignments.
       In the case an instance has a degree of membership larger than fuzzfactor for more
       than maxlabels clusters, only the assignments with the clusters with largest
       membership will be considered.
    """

    args = line.strip().split()
    (subcmd, *args) = args
    subcmd = subcmd.lower()

    try:

      if(subcmd == 'runs'):
        self.essayruns = int(args[0])
        if(self.verbose):
          print('-- number of essay runs set to {0}'.format(self.essayruns))

      elif(subcmd == 'gridcols'):
        self.gridCols = int(args[0])
        if(self.verbose):
          print('-- gridCols set to {0}'.format(self.gridCols))

      elif(subcmd == 'scalecols'):
        self.scaleCols = int(args[0])
        if(self.verbose):
          print('-- scaleCols set to {0}'.format(self.scaleCols))

      elif(subcmd == 'topn'):
        self.topN = int(args[0])
        if(self.verbose):
          print('-- topN set to {0}'.format(self.topN))

      elif(subcmd == 'filename'):

        if(len(args) > 0):
          # parses filename, extracts filetype to define targetpath
          filename = args[0]
          filetype = os.path.basename(filename).split('.')[1]
          if(filetype.lower() in ['tex', 'pgf']):
            targetpath = self.targetpath + ['tex']
          elif(filetype.lower() in ['png', 'pdf', 'jpg']):
            targetpath = self.targetpath + ['figures']
          else:
            targetpath = self.targetpath

          # updates self.filename
          if(not exists(join(*targetpath))): makedirs(join(*targetpath))
          self.filename = join(*targetpath, filename)

        else:
          self.filename = None

        if(self.verbose):
          print('-- filename set to {0}'.format(self.filename))

      elif(subcmd == 'noun'):
        option = args[0]
        self.class_noun = option
        if(self.verbose):
          print('-- noun set to {0}'.format(self.class_noun))

      elif(subcmd == 'norm'):
        option = args[0]
        if(option in ['TwoSlopeNorm', 'SymLogNorm']):
          layout_data['cbar.norm'] = option
          if(self.verbose):
            print('-- norm set to {0}'.format(option))
        else:
          print('-- unrecognised norm {0}'.format(option))

      elif(subcmd == 'cmap'):
        option = args[0]
        if(option in ['diverging', 'uniform']):
          layout_data['cbar.cmap'] = option
          if(self.verbose):
            print('-- cmap set to {0}'.format(option))
        else:
          print('-- unrecognised colour map {0}'.format(option))

      elif(subcmd == 'alpha'):
        try:
          option = float(args[0])
          self.alpha = option
          if(self.verbose):
            print('-- alpha set to {0}'.format(option))
        except:
          print('-- unrecognised confidence level {0}'.format(args[0]))

      elif(subcmd == 'testfrac'):
        try:
          option = float(args[0])
          self.testfrac = option
          if(self.verbose):
            print('-- testfrac set to {0}'.format(option))
        except:
          print('-- unrecognised test fraction {0}'.format(args[0]))

      elif(subcmd == 'rcond'):
        try:
          option = args[0]
          if(option == 'None'):
            self.rcond = None
            msg = option
          else:
            # a float value represents the fraction of the largest singular value
            # that will be used as a threshold for the inclusion of singular vectors
            # in np.linalg.lstsq
            value = float(option)
            if(0. <= value < 1. or value == -1.):
              self.rcond = value
              msg = f'{self.rcond} (corresponding to a fraction of the first singular value)'
            else:
              # sets a value so that this many singular vectors are used in lstsq
              value = int(option)
              if(value >= 1):
                self.rcond = self.model.get_rcond_by_nsvs(value)
                msg = f'{self.rcond} (corresponding to {option} singular vectors)'
              else:
                raise ValueError('Integer cutoffs must be larger than zero, got {value}')
          if(self.verbose):
            print('-- rcond set to {0}'.format(msg))
        except:
          print('-- unrecognised cutoff for singular values: {0}'.format(args[0]))

      elif(subcmd == 'fuzzfactor'):
        try:
          value = float(args[0])
          self.fuzzfactor = value
          if(self.verbose):
            print('-- fuzzfactor set to {0}'.format(value))
        except:
          print('-- unrecognised value for fuzzfactor: {0}'.format(args[0]))

      elif(subcmd == 'maxlabels'):
        try:
          value = eval(args[0])
          if(isinstance(value, int) or value is None):
            self.maxlabels = value
            if(self.verbose):
              print('-- maxlabels set to {0}'.format(value))
          else:
            print('-- unrecognised value for maxlabels: {0}'.format(args[0]))
            raise ValueError
        except:
          print('-- unrecognised value for maxlabels: {0}'.format(args[0]))

      elif(subcmd == 'vorder'):
        try:
          value = args[0]
          self.polygrid['vorder'] = value
          if(self.verbose):
            print('-- vorder set to {0}'.format(value))
        except:
          print('-- unrecognised value for vorder: {0}'.format(args[0]))

      elif(subcmd == 'solver'):
        try:
          value = args[0]
          self.polygrid['solver'] = value
          if(self.verbose):
            print('-- solver set to {0}'.format(value))
        except:
          print('-- unrecognised value for solver: {0}'.format(args[0]))

      elif(subcmd == 'corder'):
        try:
          value = args[0]
          self.corder = value
          if(self.verbose):
            print('-- class ordering set to {0}'.format(value))
        except:
          print('-- unrecognised value for class ordering: {0}'.format(args[0]))

      elif(subcmd == 'cutoff'):
        value = args[0]
        if(value in [ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE]):
          self.cutoff = value
          if(self.verbose):
            print('-- cutoff set to {0}'.format(value))
        else:
          print('-- unrecognised value for cutoff type: {0}'.format(args))
          raise ValueError

      elif(subcmd == 'sector'):
        try:
          value = args[0]
          self.polygrid['sector_type'] = value
          if(self.verbose):
            print('-- sector type set to {0}'.format(value))
        except:
          print('-- unrecognised value for sector type: {0}'.format(args[0]))

      elif(subcmd == 'annulus'):
        try:
          value = args[0]
          self.polygrid['annulus_type'] = value
          if(self.verbose):
            print('-- annulus type set to {0}'.format(value))
        except:
          print('-- unrecognised value for annulus type: {0}'.format(args[0]))

      elif(subcmd == 'size'):

        if(self.state < ECO_SPLIT):
          print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
          self.increase_exc()
          return False

        try:
          value = args[0]
          if(value.lower() in ['free', 'narrow']):
            value = value.lower()
          elif(int(value) > 0):
            value = int(value)
          (_, d) = self.X.shape
          (_, n) = self.Y.shape
          self.model_size = (value, n, d)
          if(self.verbose):
            print('-- model size set to {0}'.format(value))
        except:
          print('-- unrecognised value for model size: {0}'.format(args[0]))

      else:
        print('-- unknown sub-command: {0}'.format(subcmd))

    except:
      self.print_exc()

  def do_toggle(self, line):
    """Toggles environment variables, e.g.,
    -- toggle transpose  (toggles the orientation of Polygrid diagrams)
    -- toggle verbose    (toggles the verbose config)
    -- toggle prototypes (toggles the display of prototypes in offer/cross diagrams)
    -- toggle caseids    (toggles the use of internal rowid or external case IDs)
    """

    args = line.strip().split()
    (subcmd, *args) = args
    subcmd = subcmd.lower()

    try:

      if(subcmd == 'transpose'):
        self.transpose = not self.transpose
        if(self.verbose):
          print('-- transpose set to {0}'.format(self.transpose))

      elif(subcmd == 'verbose'):
        self.verbose = not self.verbose
        if(self.verbose):
          print('-- verbose set to {0}'.format(self.verbose))

      elif(subcmd == 'prototypes'):
        self.showprotos = not self.showprotos
        if(self.verbose):
          print('-- show prototypes set to {0}'.format(self.showprotos))

      elif(subcmd == 'caseids'):
        self.usecaseids = not self.usecaseids
        if(self.verbose):
          print('-- use of case IDs set to {0}'.format(self.usecaseids))

      else:
        print('-- unknown sub-command: {0}'.format(subcmd))

    except:
      self.print_exc()

  def do_override(self, line):
    """Overrides all attribute values of an instance as specified, e.g.,
    -- override 10 1.0   (all attributes of instance 10 set to 1.0)
    -- override 10 1.0 1 (all attributes of instance 10 set to 1.0 times the prototype for class 1)
    """
    if(self.state < ECO_TRAINED):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()

      caseID  = int(args[0])
      value   = max(0.1, min(1.0, float(args[1])))
      offerID = int(args[2]) if len(args) > 2 else None

      if(offerID is None):
        # overrides all attributes of instance 'caseID' with 'value'
        self.dataset.data[caseID,:]  = value * np.ones((1, self.model.d))
        self.dataset.odata[caseID,:] = value * self.dataset.scaling[self.model.vo]
        if(self.verbose):
          print('-- instance {0} attributes set to {1}'.format(caseID, value))

      else:
        if(self.model_type == 'Polygrid'):
          # overrides all attributes of instance 'caseID' with 'value' * class profile
          class_profile = self.model.P[offerID, self.model.get_reverse_vorder()]
          self.dataset.data[caseID,:]  = value * class_profile
          self.dataset.odata[caseID,:] = value * class_profile * self.dataset.scaling
          if(self.verbose):
            print('-- instance {0} attributes set to {1} times prototype of class {2}'.format(caseID, value, offerID))
        else:
          print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
          self.increase_exc()
          return False

    except Exception as e:
      self.print_exc()

  def do_swap(self, line):
    """Swaps a test instance with a training instance, e.g.,

    To replace test instance 80 by instance 90, which is allocated to the training partition
    -- swap 80 90

    Note that the first argument must be an instance allocated to the test partition,
    and the second argument an instance from the training partition
    """
    if(self.state < ECO_SPLIT):
      print(f'-- this command is not enabled in the current state: {state2txt[self.state]}')
      self.increase_exc()
      return False

    try:
      args = line.strip().split()

      te_idx = int(args[0])
      tr_idx = int(args[1])

      pos_te_idx = self.te_idxs.tolist().index(te_idx)
      pos_tr_idx = self.tr_idxs.tolist().index(tr_idx)

      self.tr_idxs[pos_tr_idx]=te_idx
      self.te_idxs[pos_te_idx]=tr_idx

      self.reset(ECO_SPLIT)

    except Exception as e:
      self.print_exc()
      self.reset(ECO_SPLIT)

  #-------------------------------------------------------------------------------------
  # specifies commands/features that implement an API for PolygridCLI
  #-------------------------------------------------------------------------------------

  def expand(self, line, params):
    """Expands a string with commands according to a named variable, e.g.

    The template script stored in self.register is expanded as if in a loop:
    -- exec self.register['datasets'] = ['iris', 'penguins']
    -- exec self.register['script'] = "load `self.register['datasets'][{i}]`; assign o; split; train 2 2; set filename output-2s-{i:02d}a; show scales;"
    -- exec self.register['script'] = self.expand(self.register['script'], ('i', 0, 1))
    -- exec self.API(self.register['script'], reset=False)
    """
    # https://stackoverflow.com/questions/2785755/how-to-split-but-ignore-separators-in-quoted-strings-in-python
    (varname, lb, ub) = params
    content=[]
    for i in range(lb, ub+1):
      buffer = re.sub(r'{{{0}}}'.format(varname), f'{i}', line)

      # finds and replaces /{<named variable>}/ and /{<named variable>:<numeric format>}/ patterns
      try:
        fmt = re.search(r'{{{0}:(.*?)}}'.format(varname), buffer).group(1)
        val = ('{0:' + fmt + '}').format(i)
        buffer = re.sub(r'{{{0}:(.*?)}}'.format(varname), val, buffer)
      except:
        pass

      # finds, evaluates, and replaces /`<python expression>`/ patterns
      try:
        groups = re.findall(r'`(.*?)`', buffer)
        for group in groups:
          buffer = buffer.replace('`' + group + '`', eval(group))
      except:
        pass

      content.append(buffer.rstrip(ECO_INSTR_SEP))

    return ECO_INSTR_SEP.join(content)

  def API(self, line, reset=True):
    """Runs a sequence of commands encoded in a single string
    -- commands must be separated by ECO_INSTR_SEP
    -- this serves as an API to expose PolygridCLI instances running on a Celery server
    """

    if(reset):
      self.reset(ECO_VOID)
    try:
      # parses the line, but handles the occurrence of the separator
      # inside strings declared in the script
      cmd_pttrn = re.compile(ECO_INSTR_SPLITTER)
      self.cmdqueue = cmd_pttrn.split(line)[1::2]
      stop = None
      while(not stop and len(self.cmdqueue) > 0):
        line = self.cmdqueue.pop(0)
        line = self.precmd(line)
        stop = self.onecmd(line)
        stop = self.postcmd(stop, line)

    except Exception as e:
      self.print_exc()

  def do_run(self, line):
    """Recovers Polygrid CLI script and runs it, e.g.,

    Loads and executes Polygrid CLI commands in the file ../scripts/autotest.in
    -- run ../scripts/autotest.in

    In such scripts, everything that appears after the quit command is ignored,
    including the quit command. The file must observe some encoding requirements:

    -- the text file must be is UTF-8 encoded
    -- there is a single CLI command per line
    -- lines are ended either by a \r\n or \n sequence

    The script does not allow for loops, but these can be emulated:

    -- exec self.register['mini-script'] = 'p ({i}**2)'
    -- exec self.register['mini-script'] = self.expand(self.register['mini-script'], ('i', 1, 10))
    -- exec self.API(self.register['mini-script'], reset=False)

    """
    # checks if line specifies a files
    filename = line
    if(os.path.exists(filename)):
      # recovers the content of the file and runs the script
      script = loadAsText(filename)
      script = script.replace('\r\n', ECO_INSTR_SEP) # windows
      script = script.replace('\n',   ECO_INSTR_SEP) # linux
      try:
        pos = re.search(r'(;|^)\s*quit\s*(;|$)', s).start()
        script = script[:pos]
      except:
        pass
      self.API(script)
    else:
      print(f'Not a valid file/path: {line}')

    return None

def main(configfile):

  PolygridCLI(configfile=configfile).cmdloop()
  return None

if(__name__ == '__main__'):

  main(sys.argv[1])
