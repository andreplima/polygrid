"""
 Polygrid CLI is a line-oriented command interpreter built on Python's Cmd class.
 It is a research prototype that was created to automate recurring tasks during the
 development of the Polygrid model, which was introduced in the following article:

 [1] Andre Paulino de Lima, Paula Castro, Suzana Carvalho Vaz de Andrade,
    Rosa Maria Marcucci, Ruth Caldeira de Melo, Marcelo Garcia Manzato.
    An interpretable recommendation model for psychometric data, with an application
    to gerontological primary care. 2026. Available at https://arxiv.org/abs/2601.19824
    (also at ./literature/polygrid_thesis.pdf)

 Polygrid is a transparent, interpretable recommendation model that displays an
 interactive diagram as a visual explanation for any recommendation it provides.
 For a crash course on Polygrid model/CLI, consider watching some of these videos:

 - I want a 10-min intro to Polygrid
  (soon)

 - I want to watch a hands-on presentation of the paper (30-min)
  (soon)

 - I want to explore my dataset within Polygrid CLI
  (soon)

 Please cite the article if this software turns out to be useful for your research.

 Disclaimer: The first author provides this software "as is", with no express or
 implied warranties. Also, the healthcare datasets mentioned in the paper/thesis
 above do not come with the software: users must request them from their owners
 or curators and preprocess them using the corresponding “read*.py” script.

"""

import re
import os
import sys
import time

DELAY_SHORT = .1
DELAY_LONG  = .4

if(sys.platform == 'win32'):
  import ctypes
  from ctypes import wintypes
else:
  import termios

# https://www.asciiart.eu/text-to-ascii-art
# font ANSI Pagga

logos = [

  """
                                {0}
                                {1}
                                {2}""",

  """
░█▀█░█▀█░█░░█░█░█▀▀░█▀▄░▀█▀░█▀▄ {0}
░█▀▀░█░█░█░░░█░░█░█░█▀▄░░█░░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░░▀▀▀░▀░▀░▀▀▀░▀▀░ {2}""",

  """
░█▀█░█▀█░█░░█░█░█▀▀░█▀▄░▀█▀░█▀▄ {0}
░█▀▀░█░█░█░░░█░░█░█░█▀▄░░█░░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░░▀▀▀░▀░▀░▀▀▀░▀▀░ {2}""",

  """
░█▀█░█▀█░█░░█░█░█▀▀░█▀▄░▀█░█▀▄ {0}
░█▀▀░█░█░█░░░█░░█░█░█▀▄░░█░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░░▀▀▀░▀░▀░▀▀░▀▀░ {2}""",

  """
░█▀█░█▀█░█░░█░█░█▀▀░█▀░▀█░█▀▄ {0}
░█▀▀░█░█░█░░░█░░█░█░█▀░░█░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░░▀▀▀░▀░░▀▀░▀▀░ {2}""",

  """
░█▀█░█▀█░█░░█░█░█▀░█▀░▀█░█▀▄ {0}
░█▀▀░█░█░█░░░█░░█░░█▀░░█░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░░▀▀░▀░░▀▀░▀▀░ {2}""",

  """
░█▀█░█▀█░█░░█░░█▀░█▀░▀█░█▀▄ {0}
░█▀▀░█░█░█░░░█░█░░█▀░░█░█░█ {1}
░▀░░░▀▀▀░▀▀▀░▀░▀▀░▀░░▀▀░▀▀░ {2}""",

  """
░█▀█░█▀░█░░█░░█▀░█▀░▀█░█▀▄ {0}
░█▀▀░█░░█░░░█░█░░█▀░░█░█░█ {1}
░▀░░░▀▀░▀▀░░▀░▀▀░▀░░▀▀░▀▀░ {2}""",

  """
░█▀░█▀░█░░█░░█▀░█▀░▀█░█▀▄  {0}
░█▀░█░░█░░░█░█░░█▀░░█░█░█  {1}
░▀░░▀▀░▀▀░░▀░▀▀░▀░░▀▀░▀▀░  {2}""",

  ]

def get_cursor_pos():
  # from https://stackoverflow.com/questions/35526014/how-can-i-get-the-cursors-position-in-an-ansi-terminal
  if(sys.platform == 'win32'):
    OldStdinMode  = ctypes.wintypes.DWORD()
    OldStdoutMode = ctypes.wintypes.DWORD()
    kernel32 = ctypes.windll.kernel32
    kernel32.GetConsoleMode(kernel32.GetStdHandle(-10), ctypes.byref(OldStdinMode))
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0)
    kernel32.GetConsoleMode(kernel32.GetStdHandle(-11), ctypes.byref(OldStdoutMode))
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
  else:
    OldStdinMode = termios.tcgetattr(sys.stdin)
    aux = termios.tcgetattr(sys.stdin)
    aux[3] = aux[3] & ~(termios.ECHO | termios.ICANON)
    termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, aux)

  try:
    aux = ''
    sys.stdout.write('\x1b[6n')
    sys.stdout.flush()
    while not (aux := aux + sys.stdin.read(1)).endswith('R'):
      True
    res = re.match(r'.*\[(?P<y>\d*);(?P<x>\d*)R', aux)
  finally:
    if(sys.platform == 'win32'):
      kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), OldStdinMode)
      kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), OldStdoutMode)
    else:
      termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, OldStdinMode)

  return (int(res.group('x')), int(res.group('y'))) if res else (-1, -1)

def set_cursor_pos(x, y):
  # from https://stackoverflow.com/questions/35526014/how-can-i-get-the-cursors-position-in-an-ansi-terminal
  # from https://en.wikipedia.org/wiki/ANSI_escape_code
  # CSI n ; m H -- "Cursor Position"
  if(sys.platform == 'win32'):
    OldStdinMode  = ctypes.wintypes.DWORD()
    OldStdoutMode = ctypes.wintypes.DWORD()
    kernel32 = ctypes.windll.kernel32
    kernel32.GetConsoleMode(kernel32.GetStdHandle(-10), ctypes.byref(OldStdinMode))
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0)
    kernel32.GetConsoleMode(kernel32.GetStdHandle(-11), ctypes.byref(OldStdoutMode))
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
  else:
    OldStdinMode = termios.tcgetattr(sys.stdin)
    aux = termios.tcgetattr(sys.stdin)
    aux[3] = aux[3] & ~(termios.ECHO | termios.ICANON)
    termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, aux)

  try:
    m = str(x)
    n= str(y)
    sys.stdout.write(f'\x1b[{n};{m}H') # CSI n ; m H
    sys.stdout.flush()
  finally:
    if(sys.platform == 'win32'):
      kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), OldStdinMode)
      kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), OldStdoutMode)
    else:
      termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, OldStdinMode)

  return None

def show_Polygrid_logo(configFile, static_logo=False, show_doctring=True):

  vrs = 'Polygrid CLI (prototype 7)'
  msg = f'Loading config from {configFile}'
  sep = len(msg)*'-'

  if (static_logo):
    logo = logos[-1]
    print(logo.format(vrs, sep, msg))

  else:
    os.system('cls' if sys.platform == 'win32' else 'clear')
    (x,y) = get_cursor_pos()
    eraser = logos[0]

    # expands the logo
    for logo in reversed(logos[1:]):
      set_cursor_pos(x,y)
      print(eraser.format('', '', ''))
      set_cursor_pos(x,y)
      print(logo.format('', '', ''))
      time.sleep(DELAY_SHORT)
    time.sleep(DELAY_LONG)

    # contracts the logo
    for logo in logos[1:]:
      set_cursor_pos(x,y)
      print(eraser.format('', '', ''))
      set_cursor_pos(x,y)
      print(logo.format('', '', ''))
      time.sleep(DELAY_SHORT)

    set_cursor_pos(x,y)
    print(logo.format(vrs, sep, msg))

    # shows the docstring
    if(show_doctring):
      print(__doc__)

  return None

