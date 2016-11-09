metrics =
  read_tsv('metrics_cv_075.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom'), ordered = T)) %>%
  select(-TRANS)

ggplot(metrics) +
  geom_jitter(aes(x = MODEL, y = PCC), width = 0.2, size = 0.1, color = 'gray') +
  geom_boxplot(aes(x = MODEL, y = PCC), width = 0.2, outlier.shape = NA) +
  scale_y_continuous(trans = 'identity')



metrics =
  read_tsv('metrics_roc_075.2.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom'), ordered = T),
         RSS = ifelse(is.na(RSS), 0, RSS),
         M = sqrt(ifelse(MCC < 0, 0, MCC) / (RSS + 1))) %>%
  select(-TRANS)


ggplot(metrics %>% filter(MCC > 0)) +
  geom_jitter(aes(x = MODEL, y = TNR), width = 0.2, size = 0.2, color = 'gray') +
  geom_boxplot(aes(x = MODEL, y = TNR), width = 0.2, outlier.shape = NA) +
  scale_y_continuous(trans = 'identity')
