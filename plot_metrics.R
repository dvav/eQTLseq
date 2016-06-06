require('ggplot2')
require('dplyr')

########################################################################################################################
########################################################################################################################

metrics = read.table('metrics.txt', stringsAsFactors = F, sep = ',', header = T)

########################################################################################################################
########################################################################################################################

metrics_flat = metrics %>%
  mutate(MCC = ifelse(is.na(MCC), 0, MCC)) %>%
  mutate(RSS = ifelse(is.na(RSS), 0, RSS)) %>%
  mutate(MCC2 = ifelse(MCC < 0, 0, MCC)) %>%
  mutate(M = (MCC2 / (RSS + 1)) ^ (1/2)) %>%
  mutate(SPARSITY = NGENES_HOT * NMARKERS_HOT + NGENES_POLY * NMARKERS_POLY) %>%
  mutate(MODEL = ifelse(MODEL %in% c('NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4', 'Binomial', 'Poisson'), MODEL, TRANS)) %>%
  mutate(MODEL = gsub('NBinomial', 'nbin', MODEL)) %>%
  mutate(MODEL = gsub('Binomial', 'bin', MODEL)) %>%
  mutate(MODEL = gsub('Poisson', 'pois', MODEL)) %>%
  mutate(MODEL = gsub('boxcox', 'bcox', MODEL)) %>%
  select(-c(NGENES, NGENES_HOT, NGENES_POLY, NMARKERS, NMARKERS_HOT, NMARKERS_POLY, TRANS)) %>%
  tidyr::gather(METRIC, VALUE, -c(MODEL, NSAMPLES, SPARSITY, SIZE, NOISE), convert = T)

########################################################################################################################
########################################################################################################################

mdls = c('nbin4', 'pois', 'bin','log', 'bcox', 'blom', 'voom', 'vst')
metric = 'M'
metrics_filt = metrics_flat %>%
  filter(METRIC == metric, MODEL %in% mdls, SIZE == 8, NOISE == 1) %>%
  group_by(METRIC, MODEL, NSAMPLES, SPARSITY, SIZE, NOISE) %>%
  arrange(desc(VALUE)) %>%
  slice(1:3) %>%
  ungroup() %>%
  mutate(MODEL = factor(MODEL, levels = mdls, ordered = T))

means = metrics_filt %>%
  group_by(METRIC, NSAMPLES, SPARSITY, SIZE, NOISE) %>%
  summarise(MEAN = mean(VALUE), SD = sd(VALUE)) %>%
  ungroup()

print(
  ggplot(data = metrics_filt, mapping = aes(x = MODEL, y = VALUE, fill = MODEL)) +
    geom_boxplot(alpha = 0.5, width = 0.5) +
    # geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed') +
    # geom_hline(data = means, aes(yintercept = MAX), linetype = 'dashed', color = 'red') +
    # geom_hline(data = means, aes(yintercept = MEAN - SD), linetype = 'dashed') +
    theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    scale_x_discrete(name = '') +
    scale_y_continuous(name = metric) +
    facet_grid(SPARSITY ~ NSAMPLES)
)
