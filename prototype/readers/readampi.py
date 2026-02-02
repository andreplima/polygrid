"""

  This script processes individual health assessment data collected for the article:

  [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
      Health profile of older adults assisted by the Elderly Caregiver Program of
      the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
      18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

  so it can be loaded to and handled by the Polygrid CLI environment. We thank the
  authors for sharing a sample of the collected data. This script converts participant
  answers into AMPI/AB domain scores according to guidance laid down in:

  [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
      Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
      (see Table 5, and Figure 2)

  This script creates five datasets, based on the data just described. The first one is
  a multiclass dataset in which each participant is assigned to a class determined by
  two decision boundaries described in the article, which classify patients into healthy,
  pre-fragile, or fragile individuals. The AMPI-AB-data dataset has 510 instances.

  Another dataset is a multilabel dataset created by selecting the instances in the
  sample that record referrals made to the patient. This dataset has 11 labels, which
  represent expert judgment. The AMPI-AB-ml-11 dataset has the following distributional
  characteristics:

        instances          128
        labels              11
        labelsets           15
        single labelsets     8
        cardinality       1.08
        density           0.10

  The other three datasets have synthetic assignments. Two are label ranking datasets
  derived from the first dataset, with 11 and 22 labels (AMPI-AB-lr-11, AMPI-AB-lr-22),
  each with 510 instances. The remaining one is a multilabel dataset which also is a
  copy of the first dataset, but with 22 labels (AMPI-AB-ml-22).

  The datasets with 22 labels (for multilabel or label ranking) aim to simulate charac-
  teristics of the dataset described in Table 3 of the following article:

  [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
      Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
      the WHO integrated care for older people (ICOPE) programme in clinical practice:
      a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
      https://doi.org/10.1016/S2666-7568(22)00097-6

      ICOPE-FR dataset; these values are inferred from data in Table 3:
        instances          958
        labels              22
        labelsets            ?
        single labelsets     ?
        cardinality       4.54
        density           0.21

  These datasets are created as Bunch objects (same API used in scikit-learn) and are
  stored in the following files:

  - AMPIAB-data.pkl
    This stores a multiclass datast, with 510 instances. It has the domains of the
    AMPI-AB instrument as features, sex, age, and race as factors, and the final
    classification as the multiclass target (Fragile, Semi-fragile, and Healthy).

  - AMPIAB-ml-11.pkl
    This stores a multilabel dataset with 128 instances. It has the same features and
    factors as the previous one, but its 11 binary labels are original to the data,
    corresponding to internal and external referrals made to the participant.

  - AMPIAB-ml-22.pkl
    This stores a multilabel dataset with 510 instances. It has the same features and
    factors as the first one, but its 22 binary labels were obtained from applying fuzzy
    clustering to the feature data.

  - AMPIAB-lr-11.pkl (with 11 labels)
  - AMPIAB-lr-22.pkl (with 22 labels)
    These store label ranking datasets with 510 instances, with the same features and
    factors as the first one, but its labels were obtained from applying fuzzy clustering
    to the feature data.
"""

import re
import sys
import numpy as np
import pyreadstat as pr

from os            import listdir, makedirs, remove
from os.path       import join, exists
from customDefs    import getMountedOn, setupEssayConfig, getEssayParameter
from customDefs    import tsprint, stimestamp, saveAsText, saveLog, serialise
from datasets      import compute_split_stats
from readerDefs    import createView, build_factor_matrix, build_feature_matrix
from readerDefs    import updateAssignments, createMetaDict
from sklearn.utils import Bunch

from customDefs    import ECO_DEFICIT
from datasets      import ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R
from datasets      import ECO_DB_MULTILABEL, ECO_DB_LABELRANK

def get_factor_labels(row, meta, factor_names, misses):
  res = []
  for variable in factor_names:
    value = row[variable]
    try:
      label = meta.variable_value_labels[variable][value]
    except KeyError:
      if((variable, str(value)) not in misses):
        misses.append((variable, str(value)))
      label = 'missing'
    res.append(label)
  res = ''.join([f'/{label.lower()}' for label in res]) + '/'
  return res

def build_feature_matrix_row(df, domains, domain):
  return df[domains[domain]].sum(axis=1).to_numpy().astype(np.float64)

def convert2Bunch_multiclass(df, meta, descr, subset, domains):

  (factors, features, outcomes, extra) = subset
  caseIDs = df.index.to_numpy().tolist()
  outcome = 'Classificacao_final'

  # builds the factor matrix Z
  (factor_names, Z, misses) = build_factor_matrix(df, meta, factors, get_factor_labels)
  for (variable, value) in misses:
    tsprint(f'** Unexpected level ({value}) for factor {variable}')

  # builds the feature matrix X
  (feature_names, X) = build_feature_matrix(df, domains, build_feature_matrix_row)

  # builds the target matrix Y
  n = df[outcome].unique().size
  target_names = [meta.variable_value_labels[outcome][i] for i in range(n)]
  Y = df[outcome].to_numpy().astype(int)
  assert np.isnan(Y).sum() == 0

  # builds the dataset
  dataset = Bunch(
      caseIDs = caseIDs,
      data    = X,
      target  = Y,
      factors = Z,
      feature_names = feature_names,
      target_names  = target_names,
      factor_names  = factor_names,
      symbol  = ECO_DEFICIT,
      DESCR   = descr,
  )

  # creates a report showing characteristics of the new dataset
  report = []
  (nrows, ncols) = dataset.data.shape
  report.append(f'feature matrix has {nrows} rows and {ncols} columns')
  report.append(f'feature names are {dataset.feature_names}')
  report.append(f'target  names are {dataset.target_names}')
  report.append(f'factor  names are {dataset.factor_names}')

  return (dataset, report)

def convert2Bunch_multilabel(df, meta, descr, subset, domains):

  (factors, features, outcomes, extra) = subset

  # removes the rows with no referral
  ds = df.loc[(df['ENC_INTERNOS'] != '7') | (df['ENC_EXTERNOS'] != '9')]
  caseIDs = ds.index.to_numpy().tolist()

  # builds the factor matrix Z
  (factor_names, Z, misses) = build_factor_matrix(ds, meta, factors, get_factor_labels)
  for (variable, value) in misses:
    tsprint(f'** Unexpected level ({value}) for factor {variable}')

  # builds the feature matrix X
  (feature_names, X) = build_feature_matrix(ds, domains, build_feature_matrix_row)

  # builds the target matrix Y
  # Y is encoded as a multilabel matrix, whose labels originate from the outcomes
  # encoded in the sources list. In this mapping, we decided to ignore the labels
  # that encode to 'no referral was informed'
  m = len(caseIDs)
  sources = [
            #( source field,  label that encodes that no referral was informed),
             ('ENC_INTERNOS', 'Não informado'),
             ('ENC_EXTERNOS', 'Não informado'),
            ]

  _targets      = []
  _target_names = []
  for (outcome, label_NI) in sources:

    #-- goes thru the outcome values and feeds the (partial) target matrix Y
    labels = ds[outcome].unique().tolist()
    n = len(labels)
    target_names = [meta.variable_value_labels[outcome][labels[j]] for j in range(n)]
    Y = np.zeros((m,n), dtype=int)
    for (i, value) in enumerate(ds[outcome]):
      j = labels.index(value)
      Y[i,j] = 1

    # -- removes the label that encodes 'no referral informed'
    p = target_names.index(label_NI)
    target_names.pop(p)
    Y = np.delete(Y, obj=p, axis=1)
    assert np.isnan(Y).sum() == 0

    #-- stores partial results so they can be adjointed later ...
    _target_names += target_names
    _targets.append(Y)

  # -- ... by later we meant now!
  Y = np.hstack(_targets)
  target_names = _target_names
  assert Y.sum() == 138
  assert Y.shape == (128, 11)
  assert all(Y.sum(axis=0) > 0)

  # builds the dataset
  dataset = Bunch(
      caseIDs = caseIDs,
      data    = X,
      target  = Y,
      factors = Z,
      feature_names = feature_names,
      target_names  = target_names,
      factor_names  = factor_names,
      symbol  = ECO_DEFICIT,
      DESCR   = descr,
  )

  # creates a report showing characteristics of the new dataset
  scenario = ECO_DB_MULTILABEL
  (cardinality, label_counts, imbalance, labelsets, single_labelsets, density) = \
     compute_split_stats(Y, scenario)
  (nrows, ncols) = dataset.data.shape

  report = []
  report.append(f'feature matrix has {nrows} rows and {ncols} columns')
  report.append(f'feature names are {dataset.feature_names}')
  report.append(f'target  names are {dataset.target_names}')
  report.append(f'factor  names are {dataset.factor_names}')
  report.append(f'cardinality .....: {cardinality:2.2f}')
  report.append(f'class imbalance .: {imbalance:2.2f}')
  report.append(f'#labelsets ......: {len(labelsets):4d}')
  report.append(f'#single labelsets: {len(single_labelsets):4d}')
  report.append(f'density .........: {density:2.2f}')

  return (dataset, report)

def main(configfile):

  tsprint(f'Loading parameters from {configfile}')

  setupEssayConfig(configfile)
  sourcepath  = getEssayParameter('PARAM_AMPIAB_SOURCEPATH')
  targetpath  = getEssayParameter('PARAM_AMPIAB_TARGETPATH')
  filename    = getEssayParameter('PARAM_AMPIAB_FILENAME')
  fuzzfactors = getEssayParameter('PARAM_AMPIAB_FUZZFACTOR')
  basename    = filename.split('.')[0].lower()

  # ensures the target folder is available and empty
  if(exists(join(*targetpath))):
    for f in listdir(join(*targetpath)):
      remove(join(*targetpath, f))
  else:
    makedirs(join(*targetpath))

  tsprint(f'Reading dataset file {filename} from {join(*sourcepath)}')
  (df, meta) = pr.read_sav(join(*sourcepath, filename))

  # reproduces a couple of statistics published in the abstract of [1]
  tsprint(__doc__)
  tsprint('-- comparing some dataset statistics with the published data:')
  tsprint('  ---------- statistic ----------  \tArticle  Sample')
  tsprint('  # of participants ............. :\t  {0:>5d}\t  {1:>5d}'.format(535, len(df.index)))
  tsprint('  % of women .................... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(77.6, 100 * (1 - df['sexo'].mean())))
  tsprint('  average age ................... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(76.2, df['idade_idoso_dia_aval'].mean()))

  tsprint('  P02. negative self-rated health :\t  {0:2.1f}%\t  {1:2.1f}%'.format(67.8, 100 * df['P2_auto_percepcao'].mean()))
  tsprint('  P14. difficulties in IADLs .... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(68.4, 100 * df['P14_aivds'].mean()))
  tsprint('  P05. polypharmacy ............. :\t  {0:2.1f}%\t  {1:2.1f}%'.format(58.1, 100 * df['P5_medicamentos'].mean()))
  tsprint('  P11. memory-related complaints  :\t  {0:2.1f}%\t  {1:2.1f}%'.format(55.8, 100 * df['P11_cognicao'].mean()))
  tsprint('  P04. multiple morbidities ..... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(50.6, 100 * (df['P4_condicoes_cronicas'] == 2).mean()))

  # recovers the metadata dictionary of the dataset
  tsprint('')
  tsprint('Recovering the metadata dictionary')
  filename = f'{basename}-meta.csv'
  content  = createMetaDict(meta)
  saveAsText('\n'.join(content), join(*targetpath, filename))
  tsprint(f'-- metadata dictionary saved as {join(*targetpath, filename)}')

  # creates a view with the columns that will be used ahead in pipeline
  tsprint('')
  tsprint('Creating the data view (only fields/columns what will be used)')
  # this pattern will select a questionaire item all of its subordinates
  # -- this is needed because we compute domain scores as sumscores of
  #    the subordinate items
  pattern  = re.compile('^P[0-9]')
  features = [variable for variable in meta.column_names if pattern.match(variable)]
  factors  = ['sexo', 'raca_cor', ]
  outcomes = ['Pontuacao_final', 'Classificacao_final', 'ENC_INTERNOS', 'ENC_EXTERNOS']
  extra    = []
  subset   = (factors, features, outcomes, extra)
  filename = f'{basename}-view.csv'
  (content, df) = createView(df, subset)
  saveAsText('\n'.join(content), join(*targetpath, filename))
  tsprint(f'-- selected factors : {factors}')
  tsprint(f'-- selected features: {features}')
  tsprint( '   (although P1_idade is a factor, it will be processed as a feature for now)')
  tsprint(f'-- selected outcomes: {outcomes}')
  tsprint(f'-- view with selected columns saved as {join(*targetpath, filename)}')
  tsprint(f'-- view has {len(df.index)} rows')

  # creates the mapping between domains and features
  # -- this mapping is based on data from Table 5 and Figure 2 of [2]
  tsprint('')
  tsprint('Creating the map linking domains to their related questionnaire items')
  domains = {
             'déficit cognitivo'     : ['P11a_esquecido',                # sums to   579
                                        'P11b_esquecimento_piora',
                                        'P11c_esquecimento_interesse',
                                       ],

             'déficit em ABVDs'      : ['P13a_sair_cama',                # sums to   183
                                        'P13b_vestir',
                                        'P13c_alimentar',
                                        'P13d_banho',
                                       ],

             'déficit em AIVDs'      : ['P10c_andar_400m',               # sums to   363
                                        'P10d_sentar_levantar',
                                        'P14a_Atividades_fora',          # sums to   589
                                        'P14b_dinheiro',
                                        ],

             'déficit em saúde bucal': ['P17a_protese',                  # sums to   532
                                        'P17b_matigar',
                                        'P17c_engolir',
                                        'P17d_alimento',
                                       ],

             'morbidades'            : ['P2_auto_percepcao',             # sums to   347
                                        'P4_condicoes_cronicas',         # sums to  1054
                                        'P5_medicamentos',
                                       ],
            }                                                            # overall: 3647

  for domain in domains:
    tsprint(f'-- Domain: {domain}')
    for item in sorted(domains[domain]):
      tsprint(f'   {item:<28}\t{meta.column_names_to_labels[item]}')

  # now that the original data has been gathered, we introduce the missing factor
  # that was recovered as a feature in the data view above
  factors.append('P1_idade')

  #--------------------------------------------------------------------------------------
  # creates a multiclass dataset
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset EACH-USP/PMSP AMPI-AB Sao Paulo (AMPIAB)
   ------------------------------------------------

   This dataset contains a sample of the data that was collected for the article:

   [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

   Patient answers are converted into AMPI-AB domain scores according to guidance in:

   [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
       Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
       (see Table 5, and Figure 2)

   This is a multiclass dataset, with 510 instances. It has the domains of the AMPI-AB
   instrument as features, sex, age, and race as factors, and a classification of each
   patient into Fragile, Semi-fragile, and Healthy as the multiclass target.
  """

  tsprint('')
  tsprint(f'Creating a multiclass dataset based on {basename} data, with 3 labels')
  filename = f'{basename}-data'
  (dataset, report) = convert2Bunch_multiclass(df, meta, descr, subset, domains)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)
  assert dataset.data.sum() == 3647 # according to the dataset view

  #--------------------------------------------------------------------------------------
  # records distributional characteristics observed in [5, 6], as well as the parameters
  # we use to simulate them based on the feature data of the the AMPI-AB dataset
  #--------------------------------------------------------------------------------------
  observed  = {11: {'data': 'AMPI-AB',  'cardinality': 1.08, 'density': 0.10},
               22: {'data': 'ICOPE-FR', 'cardinality': 4.54, 'density': 0.21},
              }

  simulated = {11: {'maxcareplan': 2,    'fuzzfactor': fuzzfactors[11], },
               22: {'maxcareplan': None, 'fuzzfactor': fuzzfactors[22], },
              }

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 22 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset EACH-USP/PMSP AMPI-AB Sao Paulo (AMPIAB-ML-22)
   ------------------------------------------------------

   This dataset contains a sample of the data that was collected for the article:

   [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

   Patient answers are converted into AMPI-AB domain scores according to guidance in:

   [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
       Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
       (see Table 5, and Figure 2)

   This is a multilabel dataset with 510 instances. It has the domains of the AMPI-AB
   instrument as features, and sex, age, and race as factors. Each patient is assigned
   to at least one among 22 labels. This arrangement aims to simulate distributional
   characteristics observed in a sample of the data that was collected for the article:

   [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
       Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
       the WHO integrated care for older people (ICOPE) programme in clinical practice:
       a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
       https://doi.org/10.1016/S2666-7568(22)00097-6

       ICOPE-FR dataset; these values are inferred from data in Table 3 of the
       article:
         instances          958
         labels              22
         labelsets            ?
         single labelsets     ?
         cardinality       4.54
         density           0.21
  """
  (scenario, infix) = (ECO_DB_MULTILABEL, 'ml')
  ngroups = 22
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 11 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset EACH-USP/PMSP AMPI-AB Sao Paulo (AMPIAB-LR-11)
   ------------------------------------------------------

   This dataset contains a sample of the data that was collected for the article:

   [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

   Patient answers are converted into AMPI-AB domain scores according to guidance in:

   [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
       Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
       (see Table 5, and Figure 2)

   This is a multilabel dataset with 510 instances. It has the domains of the AMPI-AB
   instrument as features, sex, age, and race as factors. Each patient is assigned to one
   or two labels among 11 labels. This arrangement aims to simulate distributional
   characteristics observed in the subsample with recorded referrals:

       AMPIAB dataset, only instances with referrals informed by the attending healthcare
       professional:
         instances          128
         labels              11
         labelsets           15
         single labelsets     8
         cardinality       1.08
         density           0.10
  """
  (scenario, infix) = (ECO_DB_LABELRANK, 'lr')
  ngroups = 11
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 22 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset EACH-USP/PMSP AMPI-AB Sao Paulo (AMPIAB-LR-22)
   ------------------------------------------------------

   This dataset contains a sample of the data that was collected for the article:

   [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

   Patient answers are converted into AMPI-AB domain scores according to guidance in:

   [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
       Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
       (see Table 5, and Figure 2)

   This is a multilabel dataset with 510 instances. It has the domains of the AMPI-AB
   instrument as features, sex, age, and race as factors. Each patient is assigned to at
   least one among 22 labels. This arrangement aims to simulate distributional
   characteristics observed in a sample of the data that was collected for the article:

   [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
       Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
       the WHO integrated care for older people (ICOPE) programme in clinical practice:
       a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
       https://doi.org/10.1016/S2666-7568(22)00097-6

       ICOPE-FR dataset; these values are inferred from data in Table 3 of the
       article:
         instances          958
         labels              22
         labelsets            ?
         single labelsets     ?
         cardinality       4.54
         density           0.21
  """
  (scenario, infix) = (ECO_DB_LABELRANK, 'lr')
  ngroups = 22
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 11 labels (real-world assignments)
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset EACH-USP/PMSP AMPI-AB Sao Paulo (AMPIAB-ML-11)
   ------------------------------------------------------

   This dataset contains a sample of the data that was collected for the article:

   [1] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

   Patient answers are converted into AMPI-AB domain scores according to guidance in:

   [2] Andrade, SC (2019). Análise psicométrica da Avaliação Multidimensional da Pessoa
       Idosa na Atenção Básica (AMPI/AB) (Master dissertation, Universidade de São Paulo).
       (see Table 5, and Figure 2)

   This is a multilabel dataset, with 128 instances. It has the domains of the AMPI-AB
   instrument as features, and sex, age, and race as factors. Each patient is assigned
   to one or two labels among 11 labels, and these labels represent expert judgment.
   The distributional characteristics of this subsample are:

       AMPIAB dataset, only instances with referrals informed by the attending healthcare
       professional:
         instances          128
         labels              11
         labelsets           15
         single labelsets     8
         cardinality       1.08
         density           0.10

  """
  (scenario, infix) = (ECO_DB_MULTILABEL, 'ml')
  ngroups = 11
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = convert2Bunch_multilabel(df, meta, descr, subset, domains)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)
  assert dataset.data.sum() == 1026 # according to the dataset view


  # saves the execution log
  logfile = 'readampiab.log'
  tsprint('')
  tsprint('This report was saved as {0}'.format(join(*targetpath, logfile)))
  tsprint('Done.')
  saveLog(join(*targetpath, logfile))

if(__name__ == '__main__'):

  main(sys.argv[1])
