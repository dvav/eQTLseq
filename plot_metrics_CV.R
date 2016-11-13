metrics =
  read_tsv('metrics.cv.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom', 'vst'), ordered = T)) %>%
  select(-TRANS)

metrics %>%
  ggplot() +
    geom_jitter(aes(x = MODEL, y = MSE), width = 0.2, size = 0.1, color = 'gray') +
    geom_boxplot(aes(x = MODEL, y = MSE), width = 0.2, outlier.shape = NA) +
    scale_y_continuous(trans = 'identity') +
    facet_grid(GROUP~NSAMPLES)

########################################################################################################################
########################################################################################################################

metrics =
  read_tsv('metrics.roc.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom', 'vst'), ordered = T),
         RSS = ifelse(is.na(RSS), 0, RSS),
         M = sqrt(ifelse(MCC < 0, 0, MCC) / (RSS + 1))) %>%
  select(-TRANS)

metrics %>%
  filter(MCC > 0) %>%
  ggplot() +
    geom_jitter(aes(x = MODEL, y = FDR), width = 0.2, size = 0.2, color = 'gray') +
    geom_boxplot(aes(x = MODEL, y = FDR), width = 0.2, outlier.shape = NA) +
    scale_y_continuous(trans = 'identity') +
    facet_wrap(~NSAMPLES)
