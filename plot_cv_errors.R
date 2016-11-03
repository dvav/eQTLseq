errors =
  read_tsv('cv_errors_075.alt.txt', col_names = c('MODEL', 'TRANS', 'REP', 'R2', 'DEV')) %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom'), ordered = T),
         R2 = ifelse(R2 < 0, 0, R2)) %>%
  select(-TRANS)

ggplot(errors %>% filter(MODEL %in% c('nbin', 'bin', 'pois'))) +
  geom_jitter(aes(x = MODEL, y = R2), width = 0.2, size = 0.1, color = 'gray') +
  geom_boxplot(aes(x = MODEL, y = R2), width = 0.2, outlier.shape = NA)



metrics =
  read_csv('metrics_ROC.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'bcox', 'blom', 'voom'), ordered = T),
         RSS = ifelse(is.na(RSS), 0, RSS),
         M = sqrt(ifelse(MCC < 0, 0, MCC) / (RSS + 1))) %>%
  select(-TRANS)


ggplot(metrics %>% filter(NSAMPLES == 400, M > 0)) +
  geom_jitter(aes(x = MODEL, y = M), width = 0.2, size= 0.2, color = 'gray') +
  geom_boxplot(aes(x = MODEL, y = M))
