metrics = read_tsv('~/WTCHG/Projects/eQTLseq/results/simdata/metrics.txt') %>%
  mutate(M = sqrt(pmax(MCC, 0) / (RMSE_TP + 1))) %>%
  select(-NGENES, -NMARKERS) %>%
  mutate(eQTLs = NGENES_HOT * NMARKERS_HOT + NGENES_POLY * NMARKERS_POLY) %>%
  select(-NGENES_HOT, -NMARKERS_HOT, -NGENES_POLY, -NMARKERS_POLY) %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL)) %>%
  select(-TRANS) %>%
  mutate(MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox')) %>%
  mutate(MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin')) %>%
  mutate(MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin')) %>%
  mutate(MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois')) %>%
  gather(METRIC, VALUE, -MODEL, -NSAMPLES, -eQTLs, -SIZE, -NOISE, -REP, convert = T) %>%
  mutate(SIZE = factor(SIZE, levels = c(2, 4, 8), labels = c('SMALL', 'MEDIUM', 'LARGE'), ordered = T)) %>%
  mutate(OVERDISPERSION = ifelse(NOISE == 1, 'NO', 'YES')) %>%
  mutate(NOISE = ifelse(NOISE == 1, 0, NOISE)) %>%
  mutate(NOISE = factor(NOISE, levels = c(0, 2, 3, 4, 5), labels = c('ABSENT', 'TYPE I', 'TYPE II', '15%', '30%'))) %>%
  mutate(MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'logit', 'arcsin',
                                          'bcox', 'blom', 'voom', 'vst'), ordered = T))

########################################################################################################################
########################################################################################################################

local({
  metrics_filt = metrics %>%
    filter(METRIC %in% c('TPR', 'TNR', 'PPV', 'NPV', 'ACC'), OVERDISPERSION == 'YES') %>%
    mutate(METRIC = factor(METRIC, levels = c('TPR', 'TNR', 'PPV', 'NPV', 'ACC'), ordered = T))

  metrics_filt_summary = metrics_filt %>%
    group_by(MODEL, NSAMPLES, METRIC) %>%
    summarise(MEAN = mean(VALUE, na.rm = T), SD = sd(VALUE, na.rm = T), SE = SD / sqrt(n())) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(NSAMPLES, METRIC) %>%
    summarise(MEAN = mean(VALUE, na.rm = T)) %>%
    ungroup()

  print(
    ggplot() +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      geom_jitter(data = metrics_filt,
                  aes(x = MODEL, y = VALUE), color = 'lightgrey', height = 0, width = 0.05, size = 0.001) +
      geom_pointrange(data = metrics_filt_summary,
                      aes(x = MODEL, y = MEAN, ymax = MEAN + 3*SE, ymin = MEAN - 3*SE),
                      color = 'black', size = 0.25, fill = 'white') +
      theme_classic() +
      theme(axis.line = element_line(size = 0.25),
            axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
            strip.background = element_blank(),
            legend.position = 'none') +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = '') +
      # scale_fill_grey(start = 0, end = 1) +
      facet_grid(METRIC~NSAMPLES, labeller = label_both, scales = 'free_y')
  )
})

########################################################################################################################
########################################################################################################################

make_plot = function(ds, vr) {
  ds_summary = ds %>%
    group_by_('MODEL', 'NSAMPLES', vr) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    ungroup() %>%
    group_by_('NSAMPLES', vr) %>%
    mutate(RANK = as.factor(length(unique(MODEL)) - rank(MEAN) + 1)) %>%
    ungroup()

  means = ds %>%
    group_by_('NSAMPLES', vr) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  ggplot() +
    geom_jitter(data = ds, aes(x = MODEL, y = VALUE),
                color = 'grey', width = 0.05, size = 0.01) +
    geom_pointrange(data = ds_summary, aes(x = MODEL, y = MEAN,
                                           ymax = MEAN + 3*SE, ymin = MEAN - 3*SE),
                    color = 'black', size = 0.25) +
    geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
    theme_classic() +
    theme(axis.line = element_line(size = 0.25),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          strip.background = element_blank(),
          legend.position = 'none') +
    scale_x_discrete(name = '') +
    scale_y_continuous(name = 'Matthews correlation coefficient') +
    # scale_fill_grey(start = 0, end = 1) +
    facet_grid(paste(vr, '~ NSAMPLES'), labeller = label_both)
}

metric = 'MCC'

metrics %>% filter(METRIC == metric,
                   OVERDISPERSION == 'YES',
                   # SIZE == 'LARGE',
                   # NOISE == 'ABSENT',
                   # eQTLs %in% c(8, 64),
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'eQTLs')

metrics %>% filter(METRIC == metric,
                   OVERDISPERSION == 'YES',
                   # NOISE == 'ABSENT',
                   # eQTLs == 64,
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'SIZE')

metrics %>% filter(METRIC == metric,
                   OVERDISPERSION == 'YES',
                   NOISE %in% c('ABSENT', '15%', '30%'),
                   # eQTLs == 64,
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'NOISE')

metrics %>% filter(METRIC == metric,
                   # eQTLs == 64,
                   # NOISE == 'ABSENT',
                   NSAMPLES %in% c(250, 2000)) %>%
  mutate(OVERDISPERSION = factor(OVERDISPERSION, levels = c('YES','NO'), ordered = T)) %>%
  make_plot(vr = 'OVERDISPERSION')

########################################################################################################################
########################################################################################################################
