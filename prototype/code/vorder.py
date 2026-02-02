import numpy as np

from collections      import deque
from itertools        import permutations
from shapely.geometry import Polygon

class VerticesOrderer:

  def __init__(self, vorder='original', ang0=0., RAD=1.0):
    self.vorder = vorder
    self.ang0 = ang0
    self.RAD  = RAD

  def polar2coord(self, r, theta):
    x = r * np.cos(theta)  # the x-coord of the vertex that sits on the axis defined by theta
    y = r * np.sin(theta)  # the y-coord of the vertex that sits on the axis defined by theta
    return (x,y)

  def scores2coords(self, scores):
    d = len(scores)
    ra = 2 * np.pi / d  # angle between two axes, in radians
    thetas = [self.ang0+(k*ra) for k in range(d)] # angles at which domain axes are located
    L = [self.polar2coord(scores[k] * self.RAD, thetas[k]) for k in range(d)]
    return L

  def coords2poly(self, coords):
    return Polygon(coords)

  def scores2poly(self, scores):
    return self.coords2poly(self.scores2coords(scores))


  def _pullseqs(self, L, d, seq = []):
    # produces a circular list of variables (seq) in which variables are sorted according
    # to scores assigned to each pair of variables (L). The sorting is done by iteratively
    # selecting the next variable to be inserted into seq, and it stops when both ends
    # coincide.
    # Useful analogy: connecting links in a chain (left or right) until it closes.

    # -- L ~ [((i, j), coeff), ...], i and j being (indicators that identify) variables
    #      and coeff corresponds to their correlation coefficient. L is assumed to be
    #      in descending order of coeff
    # -- d is the number of variables to be sorted
    # -- seq is the sorted list of variables

    # initialises the "chain" with the pair of variables with the highest score
    if(len(seq) == 0):
      (pair, _) = L.pop(0)
      for domain in pair:
        seq.append(domain)

    # goes thru elements in L aiming to extend seq
    #print('.. seq = {0}'.format(seq))
    newL = []
    while(len(L) > 0 and len(seq) < d):

      # collects the next pair with highest score
      e = L.pop(0)
      (pair, _) = e
      (i, j) = pair

      # checks if collected pair can be used to extend seq
      seqends = (seq[0], seq[-1]) # identifies the left and right ends of seq

      if(i not in seqends and j not in seqends):
        # no, the collected pair cannot be used to extend seq
        # next decision: should we discard or reschedule the pair?
        if(i in seq or j in seq):
          # case:   one of the variables (i, j) has already been linked into seq
          # action: discards the pair (because it now leads to an invalid link)
          pass
        else:
          # case:   none of the variables (i, j) have been linked into seq
          # action: postpone the processing of the pair
          newL.append(e)

      elif(i in seqends and j not in seqends):
        if(j not in seq):
          # case:   the left  element of the pair (i) matches one of the ends of seq, and
          #         the right element of the pair (j) is free
          # action: extends seq by appending j to the proper end of seq
          if(seq[-1] == i):
            seq.append(j)
          elif(i == seq[0]):
            seq = [j] + seq
        else:
          # case:   the left  element of the pair (i) matches one of the ends of seq, and
          #         the right element of the pair (j) is NOT free
          # action: discards the pair (because it now leads to an invalid link)
          pass

      elif(j in seqends and i not in seqends):
        if(i not in seq):
          # case:   the right element of the pair (j) matches one of the ends of seq, and
          #         the left  element of the pair (i) is free
          # action: extends seq by appending i to the proper end of seq
          if(j == seq[0]):
            seq = [i] + seq
          elif(seq[-1] == j):
            seq.append(i)
        else:
          # case:   the right element of the pair (j) matches one of the ends of seq, and i is not free
          # action: discards the pair (because it now leads to an invalid link)
          pass

      elif(i in seqends and j in seqends):
        # case:   the pair (i,j) matches the ends of seq
        # action: discards the pair (because it would duplicate one variable in the list)
        pass

      #print('.. pair, seq = {0}, {1}'.format(pair, seq))

    return newL, seq

  def _getAllVOs(self, domains):
    # lists all orderings of the vertices of a polygon, excluded rotations and reflections
    # domains: a list of integers, each representing a vertex of a simple polygon

    nd = len(domains)      # number of vertices
    orderings = [domains]  # accepted orderings
    accepted  = [deque(domains), deque(reversed(domains))] # accepted orderings + reflections
    for permutation in permutations(domains):
      straight = deque(permutation)
      mirrored = deque(reversed(permutation))

      # checks if the current permutation is
      # 1. a rotation   of an already accepted ordering
      # 2. a reflection of an already accepted ordering
      rejected = False
      for candidate in [straight, mirrored]:
        for _ in range(nd):
          candidate.rotate()
          if(candidate in accepted):
            rejected = True
            break

      # accepts the candidate
      if(not rejected):
        orderings.append(list(straight))
        accepted += [straight, mirrored]

    return orderings

  def _vorder0(self, X, maxiter = np.inf): # keeps the original order of the features
    (m,d) = X.shape
    seq   = list(range(d))
    return seq

  def _vorder1(self, X, maxiter = np.inf): # highest-to-lowest positive, then lowest-to-highest negative correlations

    # helper function; returns true if pair (of domains) = (i,j) with i < j
    # (i.e., it selects elements from upper triangular covariance matrix)
    f = lambda pair: pair[0] < pair[1]

    # assumes X is a (m,d) matrix
    (m,d) = X.shape
    C = np.corrcoef(X, rowvar=False)
    seq  = []

    # processes positively correlated pairs of variables
    # note: L is sorted from highest to lowest positive correlation
    L = sorted([(idx, coef) for idx, coef in np.ndenumerate(C) if f(idx) and coef > 0.0], key=lambda e: -e[1])
    #print(L)
    first = L[0][0][0] # saves the start of the list
    iter=0
    while(len(L) > 0 and iter < maxiter):
      (L, seq) = self._pullseqs(L, d, seq)
      iter+=1

    # processes negatively correlated pairs of variables (if seq still misses variables)
    # note: now L is sorted from lowest to highest negative correlation
    #      (less negative comes first)
    if(len(seq) < d):
      #print('.. complementing vorder with negative correlations')
      L = sorted([(idx, coef) for idx, coef in np.ndenumerate(C) if f(idx) and coef <= 0.0], key=lambda e: -e[1])
      #print(L)
      iter=0
      while(len(L) > 0 and iter < maxiter):
        (L, seq) = self._pullseqs(L, d, seq)
        iter+=1

    if(len(seq) != d):
      if(len(seq) == (d-1)):
        # there is only one missing variable to complete the sequence, so let's add it
        for i in range(d):
          if(i not in seq):
            seq.append(i)
            break
      else:
        # the algorithm failed and the case requires further analysis
        print('*************************************************************')
        print(f'L ..........: {L}')
        print(f'seq ........: {seq}')
        print(f'sorted(seq) : {sorted(seq)}')
        print(f'len(seq) ...: {len(seq)}')
        print(f'd ..........: {d}')
        print('*************************************************************')
        raise RuntimeError('vorder failed to order the variables of the dataset')

    # reorganises the list so that the first pair of variables is the one with
    # highest positive correlation
    start = seq.index(first)
    seq = seq[start:] + seq[0:start]

    return seq

  def _vorder2(self, X, maxiter = np.inf): # highest squared correlation

    # helper function; returns true if pair (of domains) = (i,j) with i < j
    # (i.e., it selects elements from upper triangular matrix)
    f = lambda pair: pair[0] < pair[1]

    # assumes X is a (m,d) matrix
    (m,d) = X.shape
    C = np.corrcoef(X, rowvar=False)
    seq  = []

    # processes squared-correlated pairs of variables
    # note: L is sorted from highest to lowest correlation (positive or negative)
    L = sorted([(idx, coef**2) for idx, coef in np.ndenumerate(C) if f(idx)], key=lambda e: -e[1])
    #print(L)
    first = L[0][0][0] # saves the start of the list
    iter=0
    while(len(L) > 0 and iter < maxiter):
      (L, seq) = self._pullseqs(L, d, seq)
      iter+=1

    if(len(seq) != d):
      # the algorithm failed and the case requires further analysis
      raise RuntimeError('vorder failed to order the variables of the dataset')

    # reorganises the list so that the first pair of variables are the ones with highest positive correlation
    start = seq.index(first)
    seq = seq[start:] + seq[0:start]

    return seq

  def _vorder3(self, X, maxiter = np.inf): # highest mean average levels

    # assumes X is a (m,d) matrix
    (m,d) = X.shape
    C = np.corrcoef(X, rowvar=False)
    seq  = []

    S = X.mean(axis=0)
    L = sorted([(idx, average) for idx, average in np.ndenumerate(S)], key=lambda e: -e[1])
    seq = [idx[0] for (idx, average) in L]

    return seq

  def _vorder4(self, X, maxiter = np.inf): # highest accumulated measure

    # assumes X is a (m,d) matrix
    (m,d) = X.shape
    if(d < 8):

      # determines all vertices configurations
      orderings = self._getAllVOs(list(range(d)))

      # estimates the ave'rage measure (area) of demand instances for all configurations
      estimates = []
      ss  = max(50, int(m/5))
      for (oid, vo) in enumerate(orderings):
        sao = list(np.random.choice(list(range(m)), ss)) # sample access order
        X_ = X[sao][:, vo]
        acc = 0.0
        for i in range(ss):
          acc += self.scores2poly(X_[i]).area
        estimates.append((oid, acc))

      # selects the configuration with highest accumulated measure
      L = sorted(estimates, key=lambda e: -e[1])
      seq = orderings[L[0][0]]

    else:

      # premises:
      # - the optimal permutation is near of the one in which domains are
      #   arranged so that (positively) highly correlated domains are neighbours
      # - an effort-limited lexicographic search may be successful
      #   (see Mallows distribution in Alfaro's thesis)
      try:
      	domains = self._vorder1(X, maxiter)
      except:
      	domains = list(range(d))

      gen = permutations(domains)
      estimates = []
      ss = max(50, int(m/5))
      for _ in range(maxiter):
        vo = next(gen)
        sao = list(np.random.choice(list(range(m)), ss)) # sample access order
        X_ = X[sao][:, vo]
        acc = sum([self.scores2poly(X_[i]).area for i in range(ss)])
        estimates.append((vo, acc))

      # selects the permutation with highest accumulated measure
      L = sorted(estimates, key=lambda e: -e[1])
      seq = list(L[0][0])

    return seq

  def seq(self, X, maxiter=100):
    if(  self.vorder == 'original'):
      seq = self._vorder0(X, maxiter)
    elif(self.vorder == 'rho'):
      seq = self._vorder1(X, maxiter)
    elif(self.vorder == 'rho-squared'):
      seq = self._vorder2(X, maxiter)
    elif(self.vorder == 'averages'):
      seq = self._vorder3(X, maxiter)
    elif(self.vorder == 'measures'):
      seq = self._vorder4(X, maxiter)
    else:
      raise NotImplementedError(f'Choice of vertices ordering not implemented:{self.vorder}')
    return seq
