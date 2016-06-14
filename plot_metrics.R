require('ggplot2')
require('dplyr')

########################################################################################################################
########################################################################################################################

metrics = readr::read_csv('metrics.txt')

########################################################################################################################
########################################################################################################################

metrics = metrics %>%
  mutate(MCC = ifelse(is.na(MCC), 0, MCC)) %>%
  mutate(RSS = ifelse(is.na(RSS), 0, RSS)) %>%
  mutate(M = sqrt(ifelse(MCC < 0, 0, MCC) / (RSS + 1))) %>%
  mutate(eQTLs = NGENES_HOT * NMARKERS_HOT + NGENES_POLY * NMARKERS_POLY) %>%
  mutate(MODEL = ifelse(MODEL %in% c('NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4', 'Binomial', 'Poisson'), MODEL, TRANS)) %>%
  mutate(MODEL = gsub('NBinomial', 'nbin', MODEL)) %>%
  mutate(MODEL = gsub('Binomial', 'bin', MODEL)) %>%
  mutate(MODEL = gsub('Poisson', 'pois', MODEL)) %>%
  mutate(MODEL = gsub('boxcox', 'bcox', MODEL)) %>%
  select(-c(NGENES, NGENES_HOT, NGENES_POLY, NMARKERS, NMARKERS_HOT, NMARKERS_POLY, TRANS)) %>%
  tidyr::gather(METRIC, VALUE, -c(MODEL, NSAMPLES, eQTLs, SIZE, NOISE), convert = T) %>%
  mutate(SIZE = factor(SIZE, levels = c(2, 4, 8), labels = c('SMALL', 'MEDIUM', 'LARGE'), ordered = T)) %>%
  mutate(OVERDISPERSION = ifelse(NOISE == 1, 'NO', 'YES'), ordered = T) %>%
  mutate(NOISE = ifelse(NOISE == 1, 0, NOISE)) %>%
  mutate(NOISE = factor(NOISE, levels = c(0, 2, 3, 4), labels = c('ABSENT', 'TYPE I', 'NA', 'TYPE II'))) %>%
  filter(NOISE != 'NA')

########################################################################################################################
########################################################################################################################

mdls = c('nbin4', 'pois', 'bin','log', 'bcox', 'blom', 'voom', 'vst')
metric = 'M'

########################################################################################################################
########################################################################################################################

local({
  metrics_filt = metrics %>% filter(METRIC == metric, MODEL %in% mdls, SIZE == 'MEDIUM', NOISE == 'ABSENT', OVERDISPERSION == 'YES')

  metrics_filt_summary = metrics_filt %>%
    group_by(METRIC, MODEL, NSAMPLES, eQTLs) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    arrange(desc(MEAN)) %>%
    # slice(1:3) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(METRIC, NSAMPLES, eQTLs) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = metrics_filt, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE), color = 'grey', width = 0.2, size = 0.2) +
      geom_pointrange(data = metrics_filt_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN, ymax = MEAN + SE, ymin = MEAN - SE, color = MODEL), size = 0.25, shape = 0) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      facet_grid(eQTLs ~ NSAMPLES, labeller = label_both)
  )
})

########################################################################################################################
########################################################################################################################

local({
  metrics_filt = metrics %>% filter(METRIC == metric, MODEL %in% mdls, NOISE == 'ABSENT', OVERDISPERSION == 'YES')

  metrics_filt_summary = metrics_filt %>%
    group_by(METRIC, MODEL, NSAMPLES, SIZE) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(METRIC, NSAMPLES, SIZE) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = metrics_filt, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE), color = 'grey', width = 0.2, size = 0.2) +
      geom_pointrange(data = metrics_filt_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN, ymax = MEAN + SE, ymin = MEAN - SE, color = MODEL), size = 0.25, shape = 0) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      facet_grid(SIZE~NSAMPLES, labeller = label_both)
  )
})

########################################################################################################################
########################################################################################################################

local({
  metrics_filt = metrics %>% filter(METRIC == metric, MODEL %in% mdls, SIZE == 'MEDIUM', OVERDISPERSION == 'YES')

  metrics_filt_summary = metrics_filt %>%
    group_by(METRIC, MODEL, NSAMPLES, NOISE) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(METRIC, NSAMPLES, NOISE) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = metrics_filt, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE), color = 'grey', width = 0.2, size = 0.2) +
      geom_pointrange(data = metrics_filt_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN, ymax = MEAN + SE, ymin = MEAN - SE, color = MODEL), size = 0.25, shape = 0) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      facet_grid(NOISE~NSAMPLES, labeller = label_both)
  )
})

########################################################################################################################
########################################################################################################################


local({
  metrics_filt = metrics %>% filter(METRIC == metric, MODEL %in% mdls, SIZE == 'MEDIUM', NOISE == 'ABSENT')

  metrics_filt_summary = metrics_filt %>%
    group_by(METRIC, MODEL, NSAMPLES, OVERDISPERSION) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(METRIC, NSAMPLES, OVERDISPERSION) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = metrics_filt, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE), color = 'grey', width = 0.2, size = 0.2) +
      geom_pointrange(data = metrics_filt_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN, ymax = MEAN + SE, ymin = MEAN - SE, color = MODEL), size = 0.25, shape = 0) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme(legend.position = 'none', axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      facet_grid(OVERDISPERSION~NSAMPLES, labeller = label_both)
  )
})

########################################################################################################################
########################################################################################################################
