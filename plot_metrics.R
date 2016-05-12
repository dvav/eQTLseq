require('ggplot2')
require('dplyr')

########################################################################################################################
########################################################################################################################

metrics = read.table('metrics7.txt', stringsAsFactors = F, sep = ',', header = T)

########################################################################################################################
########################################################################################################################

metrics_flat = metrics %>%
  mutate(MCC = ifelse(is.na(MCC), 0, MCC)) %>%
  mutate(RSS = ifelse(is.na(RSS), 0, RSS)) %>%
  mutate(MCC2 = ifelse(MCC < 0, 0, MCC)) %>%
  mutate(M = (MCC2 / (RSS + 1)) ^ (1/2)) %>%
  mutate(NMARKERS = NMARKERS_CAUSAL, NGENES = NGENES_AFFECTED) %>%
  mutate(MODEL = ifelse(MODEL %in% c('NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4', 'Binomial', 'Poisson'), MODEL, TRANS)) %>%
  mutate(MODEL = gsub('NBinomial', 'nbin', MODEL)) %>%
  mutate(MODEL = gsub('Binomial', 'bin', MODEL)) %>%
  mutate(MODEL = gsub('Poisson', 'pois', MODEL)) %>%
  mutate(MODEL = gsub('boxcox', 'bcox', MODEL)) %>%
  select(-c(TRANS)) %>%
  tidyr::gather(METRIC, VALUE, -c(NSAMPLES, NMARKERS, NGENES, MODEL, SIZE), convert = T)

########################################################################################################################
########################################################################################################################

mdls = c('nbin4', 'pois', 'bin', 'log', 'bcox', 'blom')
metric = 'G'
metrics_filt = metrics_flat %>%
  filter(METRIC %in% c(metric), MODEL %in% mdls, SIZE == 5) %>%
  group_by(NSAMPLES, NMARKERS, NGENES, MODEL, METRIC, SIZE) %>%
  arrange(desc(VALUE)) %>%
  slice(1:2) %>%
  ungroup() %>%
  mutate(MODEL = factor(MODEL, levels = mdls, ordered = T))

means = metrics_filt %>%
  group_by(NSAMPLES, NMARKERS, NGENES, METRIC) %>%
  summarise(MEAN = mean(VALUE)) %>%
  ungroup()

print(
  ggplot(data = metrics_filt, mapping = aes(x = MODEL, y = VALUE, fill = MODEL)) +
    geom_boxplot(alpha = 0.5, width = 0.5) +
    # geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed') +
    theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    scale_x_discrete(name = '') +
    scale_y_continuous(name = metric) +
    facet_grid(NSAMPLES ~ NMARKERS + NGENES)
)
