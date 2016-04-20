require('ggplot2')
require('dplyr')

metrics = read.table('metrics3.txt', stringsAsFactors = F, sep = ',', header = T)
metrics = metrics %>%
  tidyr::gather(METRIC, VALUE, -c(NSAMPLES, NGENES, NGENES_AFFECTED, NMARKERS, NMARKERS_CAUSAL, MODEL, TRANS)) %>%
  mutate(MODEL2 = ifelse(MODEL %in% c('NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4', 'Poisson', 'Binomial'), MODEL, TRANS)) %>%
  mutate(MODEL2 = gsub('NBinomial', 'nbin', MODEL2)) %>%
  mutate(MODEL2 = gsub('Binomial', 'bin', MODEL2)) %>%
  mutate(MODEL2 = gsub('Poisson', 'pois', MODEL2))

metrics_filt = metrics %>%
  filter(METRIC %in% c('RSS'), MODEL2 %in% c('nbin4', 'log', 'boxcox', 'blom'))

print(
  ggplot(data = metrics_filt, mapping = aes(x = MODEL2, y = VALUE, fill = MODEL2)) +
    geom_boxplot(alpha = 0.5) +
    theme(legend.position = "none") +
    facet_grid(NSAMPLES ~ NMARKERS_CAUSAL + NGENES_AFFECTED, labeller = label_both)
)
