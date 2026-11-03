"""
 Polygrid CLI is a line-oriented command interpreter built on Python's Cmd class. It is
 a research prototype that was created to automate recurring tasks during the development
 of the Polygrid model, which was introduced in the following article:

 [1] Andre Paulino de Lima, Paula Castro, Suzana Carvalho Vaz de Andrade, 
     Rosa Maria Marcucci, Ruth Caldeira de Melo, Marcelo Garcia Manzato. 
     An Interpretable Recommendation Model for Psychometric Data, 
     With an Application to Gerontological Primary Care. 2026. 
     Available at https://arxiv.org/abs/2601.19824

 Polygrid is a transparent, interpretable recommendation model that displays an
 interactive diagram as a visual explanation for any recommendation it provides.
 For a crash course on Polygrid model/CLI, consider watching some of these videos:

 - I want a 10-min intro to Polygrid
   https://player.vimeo.com/video/918537658

 - I want to watch a hands-on presentation of the paper (30-min)
   https://player.vimeo.com/video/918537658

 - I want to explore my dataset within Polygrid CLI
   https://player.vimeo.com/video/918537658

 Please cite the article if this software turns out to be useful for your research.
 Disclaimers: (1) The first author provides this software "as is", with no express 
 or implied warranties. (2) The healthcare datasets mentioned in the paper above do 
 not come with this software: users must request them from their owners or curators.
"""

import re
import sys
import time

from contextlib import contextmanager

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

@contextmanager
def animation_terminal():
  """
  Prepares the terminal for the animation and restores it on exit, even if the
  animation is interrupted: hides the cursor, enables ANSI sequences on Windows,
  and stops typed keys from being echoed over the logo. Keys typed during the
  animation are discarded on exit, so they do not reach the CLI prompt.
  """
  if(sys.platform == 'win32'):
    kernel32 = ctypes.windll.kernel32
    hStdin   = kernel32.GetStdHandle(-10)
    hStdout  = kernel32.GetStdHandle(-11)
    OldStdinMode  = ctypes.wintypes.DWORD()
    OldStdoutMode = ctypes.wintypes.DWORD()
    kernel32.GetConsoleMode(hStdin,  ctypes.byref(OldStdinMode))
    kernel32.GetConsoleMode(hStdout, ctypes.byref(OldStdoutMode))
  else:
    OldStdinMode = termios.tcgetattr(sys.stdin)

  try:
    if(sys.platform == 'win32'):
      kernel32.SetConsoleMode(hStdin,  1) # ENABLE_PROCESSED_INPUT keeps Ctrl+C working
      kernel32.SetConsoleMode(hStdout, 7) # ENABLE_VIRTUAL_TERMINAL_PROCESSING and defaults
    else:
      aux = termios.tcgetattr(sys.stdin)
      aux[3] = aux[3] & ~(termios.ECHO | termios.ICANON)
      termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, aux)

    sys.stdout.write('\x1b[?25l') # hides the cursor
    sys.stdout.flush()
    yield

  finally:
    try:
      sys.stdout.write('\x1b[?25h') # shows the cursor
      sys.stdout.flush()
    finally:
      if(sys.platform == 'win32'):
        kernel32.FlushConsoleInputBuffer(hStdin)
        kernel32.SetConsoleMode(hStdin,  OldStdinMode)
        kernel32.SetConsoleMode(hStdout, OldStdoutMode)
      else:
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, OldStdinMode)

def clear_screen():
  # Uses ANSI rather than a cls/clear subprocess, which could change the Windows
  # console modes set by animation_terminal.
  sys.stdout.write('\x1b[H\x1b[2J\x1b[3J')
  sys.stdout.flush()

  return None

def get_cursor_pos():
  # Must be called inside animation_terminal
  # from https://stackoverflow.com/questions/35526014/how-can-i-get-the-cursors-position-in-an-ansi-terminal
  aux = ''
  sys.stdout.write('\x1b[6n')
  sys.stdout.flush()
  while not (aux := aux + sys.stdin.read(1)).endswith('R'):
    True
  res = re.match(r'.*\[(?P<y>\d*);(?P<x>\d*)R', aux)

  return (int(res.group('x')), int(res.group('y'))) if res else (-1, -1)

def set_cursor_pos(x, y):
  # Must be called inside animation_terminal
  # from https://stackoverflow.com/questions/35526014/how-can-i-get-the-cursors-position-in-an-ansi-terminal
  # from https://en.wikipedia.org/wiki/ANSI_escape_code
  # CSI n ; m H -- "Cursor Position"
  m = str(x)
  n = str(y)
  sys.stdout.write(f'\x1b[{n};{m}H') # CSI n ; m H
  sys.stdout.flush()

  return None

def show_Polygrid_logo(configFile, static_logo=False, show_doctring=True):

  vrs = 'Polygrid CLI (prototype 7)'
  msg = f'Loading config from {configFile}'
  sep = len(msg)*'-'

  if (static_logo):
    logo = logos[-1]
    print(logo.format(vrs, sep, msg))

  else:
    with animation_terminal():
      clear_screen()
      (x,y) = get_cursor_pos()
      eraser = logos[0]

      # expands the logo
      for logo in reversed(logos[1:]):
        #set_cursor_pos(x,y)
        #print(eraser.format('', '', ''))
        set_cursor_pos(x,y)
        print(logo.format('', '', ''))
        time.sleep(.08)
      time.sleep(.50)

      # contracts the logo
      for logo in logos[1:]:
        #set_cursor_pos(x,y)
        #print(eraser.format('', '', ''))
        set_cursor_pos(x,y)
        print(logo.format('', '', ''))
        time.sleep(.08)

      set_cursor_pos(x,y)
      print(logo.format(vrs, sep, msg))

    # shows the docstring
    if(show_doctring):
      print(__doc__)

  return None

