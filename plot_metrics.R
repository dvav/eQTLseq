require('ggplot2')
require('dplyr')

metrics = read.table('metrics3.txt', stringsAsFactors = F, sep = ',', header = T)
metrics_flat = metrics %>%
  mutate(MCC = ifelse(is.na(MCC), 0, MCC)) %>%
  mutate(RSS = ifelse(is.na(RSS), 0, RSS)) %>%
  mutate(MCC2 = ifelse(MCC < 0, 0, MCC)) %>%
  mutate(M = (MCC2 / (RSS + 1)) ^ (1/2)) %>%
  mutate(MODEL2 = ifelse(MODEL %in% c('NBinomial', 'Binomial', 'Poisson'), MODEL, TRANS)) %>%
  mutate(MODEL2 = gsub('NBinomial', 'nbin', MODEL2)) %>%
  mutate(MODEL2 = gsub('Binomial', 'bin', MODEL2)) %>%
  mutate(MODEL2 = gsub('Poisson', 'pois', MODEL2)) %>%
  mutate(MODEL2 = gsub('boxcox', 'bcox', MODEL2)) %>%
  select(-c(MODEL,TRANS)) %>%
  tidyr::gather(METRIC, VALUE, -c(NSAMPLES, NGENES, NGENES_AFFECTED,
                                  NMARKERS, NMARKERS_CAUSAL, MODEL2, S2),
                convert = T)

mdls = c('nbin','pois', 'bin', 'log', 'bcox', 'blom')
metric = 'PPV'
metrics_filt = metrics_flat %>%
  filter(METRIC %in% c(metric), S2==2) %>%
  select(-METRIC) %>%
  group_by(NSAMPLES, NMARKERS, NMARKERS_CAUSAL, NGENES, NGENES_AFFECTED, MODEL2) %>%
  arrange(desc(VALUE)) %>%
  slice(1:3) %>%
  ungroup() %>%
  mutate(MODEL2=factor(MODEL2, levels = mdls, ordered = T))

means = metrics_filt %>%
  group_by(NSAMPLES, NMARKERS, NMARKERS_CAUSAL, NGENES, NGENES_AFFECTED) %>%
  summarise(MEAN = mean(VALUE)) %>%
  ungroup()


print(
  ggplot(data = metrics_filt, mapping = aes(x = MODEL2, y = VALUE, fill = MODEL2)) +
    geom_boxplot(alpha = 0.5, width = 0.5) +
    geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed') +
    theme(legend.position = 'none') +
    scale_x_discrete(name='') +
    scale_y_continuous(name=metric) +
    facet_wrap(NSAMPLES~NMARKERS_CAUSAL + NGENES_AFFECTED, labeller = label_both, ncol = 4)
)
