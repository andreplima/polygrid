class LayoutManager():

  def __init__(self, seed_data):
    self.seed = seed_data

  def __call__(self, context=None, **kwargs):

    (diagram, component, *args) = context.strip().split('.')

    if(component in ['title', 'title_1st']):
      # returns the font properties to format a subplot title
      fontdict = {'family': self.seed['family'],
                  'color':  self.seed['color'],
                  'weight': self.seed['weight'],
                  'size':   self.seed[f'{diagram}.{component}.size'],
                 }
      return {'fontdict': fontdict, 'pad': self.seed[f'{diagram}.{component}.pad']}

    elif(component == 'figure_sizes'):
      # returns the sizes, margins, and spacings for a figure
      data = self.seed[f'{diagram}.plot.sizes']
      (m,n) = (kwargs['nrows'], kwargs['ncols'])

      sizeh = data['bspace'] + (m-1)*data['hspace'] + m*data['unith'] + data['tspace']
      sizew = data['lspace'] + (n-1)*data['wspace'] + n*data['unitw'] + data['rspace']
      sizeh = min(sizeh, data['maxsizeh'])
      sizew = min(sizew, data['maxsizew'])

      adjusts = dict(bottom = data['bspace']/sizeh,
                     left   = data['lspace']/sizew,
                     top    = 1 - data['tspace']/sizeh,
                     right  = 1 - data['rspace']/sizew,
                     hspace=data['hspace'],
                     wspace=data['wspace'])

      #print()
      #print(f"top    {adjusts['top']:4.3f}    bottom {adjusts['bottom']:4.3f} ")
      #print(f"left   {adjusts['left']:4.3f}   right  {adjusts['right']:4.3f} ")
      #print(f"hspace {adjusts['hspace']:4.3f} wspace {adjusts['wspace']:4.3f} ")

      # increases sizew if diagram contains a colour bar
      if(diagram in ['offer', 'cross']):
        sizew = min(sizew + data['cbar'], data['maxsizew'])

      # increases sizew if diagram contains a large tag
      if(diagram in ['demand'] and n > 1):
        sizew = min(sizew + (n-1) * data['ltag'], data['maxsizew'])

      return (sizew, sizeh, adjusts)

    elif(component == 'tag'):
      # returns font properties to format a tag (annotate)
      return  {'family':   self.seed['family'],
               'color':    self.seed['color'],
               'weight':   self.seed['weight'],
               'fontsize': self.seed[f'{diagram}.tag.size'],
               'xy':       self.seed[f'{diagram}.tag.xy'],
               'xytext':   self.seed[f'{diagram}.tag.xy'],
               'zorder':   self.seed[f'{diagram}.tag.layer'],
               'ha':       self.seed[f'{diagram}.tag.ha'],
               'va':       self.seed[f'{diagram}.tag.va'],
               'xycoords': self.seed[f'{diagram}.tag.coords'],
               'annotation_clip': self.seed['clip'],
               'bbox': dict(boxstyle = 'round',
                            fc    = self.seed[f'{diagram}.tag.fc'],
                            alpha = self.seed[f'{diagram}.tag.alpha'], ),
              }

    elif(component == 'region'):
      # returns the font properties to format the annulus sectors
      return self.seed[f'{diagram}.region']

    elif(component == 'label'):
      # returns a font dictionary, offset and alignment of a dimension label
      fontdict = {'family': self.seed['family'],
                  'color':  self.seed['color'],
                  'weight': self.seed['weight'],
                  'size':   self.seed[f'{diagram}.label.size'],
                  'ha': 'center',
                  'va': 'center',
                 }
      return {'fontdict':  fontdict,
              'offset':    self.seed[f'{diagram}.label.offset'],
              'alignment': self.seed[f'{diagram}.label.alignment'],
             }

    elif(context == 'cbar.norm'):
      # returns the color norm to be used in the colour bar
      return self.seed['cbar.norm']

    elif(context == 'cbar.cmap'):
      # returns the color map to be used in the colour bar
      return self.seed['cbar.cmap']

    elif(context == 'cbar.factor'):
      # returns a font dictionary to format a factor of the colour bar
      fontdict = {'family': self.seed['family'],
                  'color':  self.seed['color'],
                  'weight': self.seed['weight'],
                  'size':   self.seed['cbar.factor.size'],
                 }
      return {'fontdict': fontdict, 'ha': 'center', 'va': 'center'}

    elif(context == 'cbar.ticks'):
      # returns a font dictionary to format the font of the colour bar ticks
      return {'labelcolor':      self.seed['color'],
              'labelsize':       self.seed['cbar.ticks.size'],
             }

    else:
      raise ValueError(f'Unexpected context for Font Manager: {context}.')

    return result
