# layout configs

# sets the point where tags touch the border of the unit disc boundary
# -- this may be useful
#    https://www.intmath.com/complex-numbers/convert-polar-rectangular-interactive.php
#touchpoint  = (0.53, 0.91) # 1.1 e^(i \pi/3)
#touchpoint  = (0.63, 0.90) # 1.1 e^(i 55o)
#touchpoint  = (0.71, 0.84) # 1.1 e^(i 50o)
touchpoint  = (0.78, 0.78) # 1.1 e^(i \pi/4)

layout_data = {# generic layout configs
               'family': 'arial',
               'color':  'black',
               'weight': 'normal',
               'clip':    False,
               'generic.size': 15,

               # summary tag layout configs
               'summary.tag.size': 15,
               'summary.tag.alpha': 0.85,
               'summary.tag.xy': (0.5, 0.65),
               'summary.tag.fc': 'lemonchiffon',
               'summary.tag.layer': 0,
               'summary.tag.ha': 'center',
               'summary.tag.va': 'center',
               'summary.tag.coords': 'data',

               # intercept tags layout configs
               'intercept.tag.size': 13,
               'intercept.tag.alpha': 0.85,
               'intercept.tag.xy': (0.0, -1.1),
               'intercept.tag.fc': 'lightgray',
               'intercept.tag.layer': 0,
               'intercept.tag.ha': 'center',
               'intercept.tag.va': 'top',
               'intercept.tag.coords': 'data',

               # demand chart/tag layout configs
               'demand.title.size': 16,
               'demand.title.pad': 0,
               'demand.title_1st.size': 16,
               'demand.title_1st.pad': 20,
               'demand.tag.size': 13,
               'demand.tag.alpha': 1.0,
               'demand.tag.xy': touchpoint,
               'demand.tag.fc': 'w',
               'demand.tag.layer': 50,
               'demand.tag.ha': 'left',
               'demand.tag.va': 'bottom',
               'demand.tag.coords': 'data',
               'demand.plot.sizes': dict(tspace=0.5,
                                         bspace=0.2,
                                         lspace=0.2,
                                         rspace=0.5,
                                         hspace=0.2,
                                         wspace=0.3,
                                         unitw=3.34,
                                         unith=2.90,
                                         cbar=0.0,
                                         ltag=0.5,
                                         maxsizew=23.0,
                                         maxsizeh=9.8),


               # offer chart/tag layout configs
               'offer.title.size': 20,
               'offer.title.pad': 0,
               'offer.title_1st.size': 20,
               'offer.title_1st.pad': 20,
               #'offer.region': dict(alpha=1.0,
               #                     ec='lightgrey',
               #                     lw=0.5),
               'offer.region': dict(alpha=1.0,
                                    ec='none',
                                    lw=0.0),
               'offer.tag.size': 13,
               'offer.tag.alpha': 0.4,
               'offer.tag.xy': touchpoint,
               'offer.tag.fc': 'w',
               'offer.tag.layer': 0,
               'offer.tag.ha': 'left',
               'offer.tag.va': 'bottom',
               'offer.tag.coords': 'data',
               'offer.plot.sizes': dict(tspace=0.5,
                                        bspace=0.2,
                                        lspace=0.2,
                                        rspace=1.5,
                                        hspace=0.2,
                                        wspace=0.3,
                                        unitw=3.34,
                                        unith=2.90,
                                        cbar=1.5,
                                        ltag=0.0,
                                        maxsizew=23.0,
                                        maxsizeh=9.8),


               # matching chart/tag layout configs
               #'cross.region': dict(alpha=0.3,
               #                     ec='none',
               #                     lw=0.0,
               #                     zorder=1),
               'cross.region': dict(alpha=1.0,
                                    ec='white',
                                    lw=6.0),
               'cross.tag.size': 13,
               'cross.tag.alpha': 0.85,
               'cross.tag.xy': touchpoint,
               'cross.tag.fc': 'lemonchiffon',
               'cross.tag.layer': 0,
               'cross.tag.ha': 'left',
               'cross.tag.va': 'bottom',
               'cross.tag.coords': 'data',
               'cross.plot.sizes': dict(tspace=0.5,
                                        bspace=0.4,
                                        lspace=0.4,
                                        rspace=1.0,
                                        hspace=0.2,
                                        wspace=0.05,
                                        unitw=3.34,
                                        unith=2.90,
                                        cbar=1.5,
                                        ltag=0.0,
                                        maxsizew=23.0,
                                        maxsizeh=9.8),


               # barsgrid chart/tag layout configs
               'bars.region': dict(alpha=0.3,
                                    ec='none',
                                    lw=0.0,
                                    zorder=1),
               'bars.tag.size': 12,
               'bars.tag.alpha': 1.0,
               'bars.tag.xy': (1.0,1.0),
               'bars.tag.fc': 'lemonchiffon',
               'bars.tag.layer': 5,
               'bars.tag.ha': 'right',
               'bars.tag.va': 'center',
               'bars.tag.coords': 'axes fraction',
               'bars.plot.sizes': dict(tspace=0.5,
                                       bspace=0.4,
                                       lspace=0.35,
                                       rspace=1.0,   # 0.0,  (disabled colour bar)
                                       hspace=0.2,
                                       wspace=0.0,   # 0.05, (disabled colour bar)
                                       unitw=3.34,
                                       unith=2.90,
                                       cbar=1.5,     # 0.0,  (disabled colour bar)
                                       ltag=0.0,
                                       maxsizew=23.0,
                                       maxsizeh=9.8),


               # disc chart layout configs
               'disc.region': dict(alpha=1.0,
                                    ec='k',
                                    lw=0.25),
               'disc.label.size': 18, #28
               'disc.label.offset': None,
               'disc.label.alignment': None,
               'disc.plot.sizes': dict(tspace=0.4,
                                       bspace=0.4,
                                       lspace=0.0,
                                       rspace=0.0,
                                       hspace=0.0,
                                       wspace=0.0,
                                       unitw=3.34*2,
                                       unith=2.90*2,
                                       cbar=0.0,
                                       ltag=0.0,
                                       maxsizew=23.0,
                                       maxsizeh=9.8),

               # inspect chart layout configs
               'scales.title.size': 16,
               'scales.title.pad': 10,
               'scales.plot.sizes': dict(
                                       tspace=0.8,
                                       bspace=0.4,
                                       lspace=0.7,
                                       rspace=0.2,
                                       hspace=0.8,
                                       wspace=0.07,
                                       unitw=12.0,
                                       unith=1.5,
                                       cbar=0.0,
                                       ltag=0.0,
                                       maxsizew=100.0,
                                       maxsizeh=9.8),

               # layout configs for offline evaluation panel
               'offlineval.title.size': 20,
               'offlineval.title.pad': 0,
               'offlineval.plot.sizes': dict(tspace=0.95, #.949
                                        bspace=0.5,       #.027
                                        lspace=1.0,       #.087
                                        rspace=0.2,       #.980
                                        hspace=0.25,
                                        wspace=0.275,
                                        unitw=3.0,
                                        unith=2.0,
                                        cbar=0.0,
                                        ltag=0.0,
                                        maxsizew=100.0,
                                        maxsizeh=100.0),

               # layout configs for user study evaluation panel 1
               'userstudy-steps.title.size': 20,
               'userstudy-steps.title.pad': 0,
               'userstudy-steps.plot.sizes': dict(tspace=0.90,
                                        bspace=0.5,
                                        lspace=0.5,
                                        rspace=0.1,
                                        hspace=0.10,
                                        wspace=0.05,
                                        unitw=6.5,
                                        unith=3.0,
                                        cbar=0.0,
                                        ltag=0.0,
                                        maxsizew=100.0,
                                        maxsizeh=100.0),

               # layout configs for user study evaluation panel 2
               'userstudy-cases.title.size': 20,
               'userstudy-cases.title.pad': 0,
               'userstudy-cases.plot.sizes': dict(tspace=0.90,
                                        bspace=0.5,
                                        lspace=0.5,
                                        rspace=0.1,
                                        hspace=0.10,
                                        wspace=0.05,
                                        unitw=13.0,
                                        unith=4.0,
                                        cbar=0.0,
                                        ltag=0.0,
                                        maxsizew=100.0,
                                        maxsizeh=100.0),

               # layout configs of supporting elements
               'dimension.label.size': 15,
               'dimension.label.offset': 1.1,
               'dimension.label.alignment': '|',

               'cbar.norm': 'TwoSlopeNorm',
               'cbar.cmap': 'diverging',

               #'cbar.norm': 'TwoSlopeNorm',
               #'cbar.cmap': 'uniform',

               #'cbar.norm': 'SymLogNorm',
               #'cbar.cmap': 'diverging',

               #'cbar.norm': 'SymLogNorm',
               #'cbar.cmap': 'uniform',

               'cbar.factor.size': 14,
               'cbar.ticks.size':  14,

             }
