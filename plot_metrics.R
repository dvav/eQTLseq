require('ggplot2')
require('dplyr')

map_mdls = list(NBinomial = 'NB', Normal = 'N', Poisson = 'P')
map_trans = list(none = 'NB', Normal = 'N', Poisson = 'P')

metrics = read.table('metrics.txt', stringsAsFactors = F, sep = ',', header = T)
metrics = metrics %>%
  tidyr::gather(METRIC, VALUE, -c(NSAMPLES, NGENES, NGENES_AFFECTED, NMARKERS, NMARKERS_CAUSAL, MODEL, TRANS)) %>%
  mutate(MODEL2 = ifelse(MODEL %in% c('NBinomial', 'Poisson'), MODEL, TRANS))

metrics_filt = metrics %>%
  filter(METRIC %in% c('PPV'), TRANS != 'scaled', NGENES_AFFECTED > 1, MODEL %in% c('Normal', 'Poisson', 'NBinomial'))

ggplot(data = metrics_filt, mapping = aes(x = MODEL2, y = VALUE, fill = MODEL2)) +
  geom_boxplot(alpha = 0.5) +
  facet_grid(NSAMPLES ~ NMARKERS_CAUSAL + NGENES_AFFECTED)
