require('ggplot2')
require('dplyr')

########################################################################################################################
########################################################################################################################

metrics = readr::read_csv('metrics.txt') %>%
  mutate(MCC = ifelse(is.na(MCC), 0, MCC)) %>%
  mutate(RSS = ifelse(is.na(RSS), 0, RSS)) %>%
  mutate(M = sqrt(ifelse(MCC < 0, 0, MCC) / (RSS + 1))) %>%
  mutate(eQTLs = NGENES_HOT * NMARKERS_HOT + NGENES_POLY * NMARKERS_POLY) %>%
  mutate(MODEL = ifelse(MODEL %in% c('NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4', 'Binomial', 'Poisson'), MODEL, TRANS)) %>%
  mutate(MODEL = gsub('NBinomial', 'nbin', MODEL)) %>%
  mutate(MODEL = gsub('nbin4', 'nbin', MODEL)) %>%
  mutate(MODEL = gsub('Binomial', 'bin', MODEL)) %>%
  mutate(MODEL = gsub('Poisson', 'pois', MODEL)) %>%
  mutate(MODEL = gsub('boxcox', 'bcox', MODEL)) %>%
  select(-c(NGENES, NGENES_HOT, NGENES_POLY, NMARKERS, NMARKERS_HOT, NMARKERS_POLY, TRANS)) %>%
  tidyr::gather(METRIC, VALUE, -c(MODEL, NSAMPLES, eQTLs, SIZE, NOISE), convert = T) %>%
  mutate(SIZE = factor(SIZE, levels = c(2, 4, 8), labels = c('SMALL', 'MEDIUM', 'LARGE'), ordered = T)) %>%
  mutate(OVERDISPERSION = ifelse(NOISE == 1, 'NO', 'YES')) %>%
  mutate(NOISE = ifelse(NOISE == 1, 0, NOISE)) %>%
  filter(NOISE %in% c(0,2,5)) %>%
  mutate(NOISE = factor(NOISE, levels = c(0, 2, 5), labels = c('ABSENT', 'TYPE I', 'TYPE II')))

########################################################################################################################
########################################################################################################################

mdls = c('nbin', 'pois', 'bin','log', 'bcox', 'blom', 'voom', 'vst')
metric = 'M'

########################################################################################################################
########################################################################################################################

local({
  metrics_filt = metrics %>% filter(METRIC == metric,
                                    MODEL %in% mdls,
                                    OVERDISPERSION == 'YES',
                                    SIZE == 'LARGE',
                                    NOISE == 'ABSENT')

  metrics_filt_summary = metrics_filt %>%
    group_by(MODEL, NSAMPLES) %>%
    summarise(MEAN = mean(VALUE), SD = sd(VALUE), SE = SD / sqrt(n())) %>%
    ungroup() %>%
    group_by(NSAMPLES) %>%
    mutate(RANK = as.factor(length(mdls) - rank(MEAN) + 1)) %>%
    ungroup()

  means = metrics_filt %>%
    group_by(NSAMPLES) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = metrics_filt, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE),
                  color = 'grey', width = 0.2, size = 0.01) +
      geom_pointrange(data = metrics_filt_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN,
                                                       ymax = MEAN + 2*SE, ymin = MEAN - 2*SE, fill = RANK),
                      color = 'black', shape = 21, size = 0.5) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), strip.background = element_blank()) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      scale_fill_grey(start = 0, end = 1) +
      facet_wrap(~NSAMPLES, labeller = label_both)
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
    mutate(RANK = as.factor(length(mdls) - rank(MEAN, ties.method = 'random') + 1)) %>%
    ungroup()

  means = ds %>%
    group_by_('NSAMPLES', vr) %>%
    summarise(MEAN = mean(VALUE)) %>%
    ungroup()

  print(
    ggplot() +
      geom_jitter(data = ds, aes(x = factor(MODEL, levels = mdls, ordered = T), y = VALUE),
                  color = 'grey', width = 0.2, size = 0.01) +
      geom_pointrange(data = ds_summary, aes(x = factor(MODEL, levels = mdls, ordered = T), y = MEAN,
                                             ymax = MEAN + 2*SE, ymin = MEAN - 2*SE, fill = RANK),
                      color = 'black', shape = 21, size = 0.5) +
      geom_hline(data = means, aes(yintercept = MEAN), linetype = 'dashed', size = 0.25) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), strip.background = element_blank()) +
      scale_x_discrete(name = '') +
      scale_y_continuous(name = metric) +
      scale_fill_grey(start = 0, end = 1) +
      facet_grid(paste(vr, '~ NSAMPLES'), labeller = label_both)
  )
}

metrics %>% filter(METRIC == metric,
                   MODEL %in% mdls,
                   OVERDISPERSION == 'YES',
                   SIZE == 'LARGE',
                   NOISE == 'ABSENT',
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'eQTLs')

metrics %>% filter(METRIC == metric,
                   MODEL %in% mdls,
                   OVERDISPERSION == 'YES',
                   NOISE == 'ABSENT',
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'SIZE')

metrics %>% filter(METRIC == metric,
                   MODEL %in% mdls,
                   OVERDISPERSION == 'YES',
                   SIZE == 'LARGE',
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'NOISE')

metrics %>% filter(METRIC == metric,
                   MODEL %in% mdls,
                   SIZE == 'LARGE',
                   NOISE == 'ABSENT',
                   NSAMPLES %in% c(250, 2000)) %>%
  make_plot(vr = 'OVERDISPERSION')

########################################################################################################################
########################################################################################################################
