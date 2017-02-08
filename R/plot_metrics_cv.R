metrics_cv = read_tsv('~/WTCHG/Projects/eQTLseq/results/geuvadis/metrics_cv.txt') %>%
  mutate(CCC = sqrt(CCC),
         MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'logit',
                                          'arcsin', 'bcox', 'blom', 'voom', 'vst'), ordered = T)) %>%
  select(-TRANS) %>%
  gather(METRIC, VALUE, CCC, RMSE, NRMSE, R2, eQTLs, n_assoc)

metrics_cv_filt = metrics_cv %>%
  filter(NSAMPLES == 339, METRIC %in% c('CCC', 'n_assoc'), GROUP %in% c('miRNAs', 'genes'))

metrics_cv_filt_summary = metrics_cv_filt %>%
  group_by(MODEL, GROUP, METRIC) %>%
  summarise(MEAN = mean(VALUE, na.rm = T), SD = sd(VALUE, na.rm = T), SE = SD / sqrt(n())) %>%
  ungroup()

means = metrics_cv_filt %>%
  group_by(GROUP, METRIC) %>%
  summarise(MEAN = mean(VALUE, na.rm = T), SD = sd(VALUE, na.rm = T), SE = SD / sqrt(n())) %>%
  ungroup()

ggplot() +
  geom_hline(aes(yintercept = MEAN), data = means, linetype = 'dashed', size = 0.25) +
  geom_jitter(aes(x = MODEL, y = VALUE), data = metrics_cv_filt, width = 0.2, size = 0.1, color = 'grey70') +
  geom_pointrange(aes(x = MODEL, y = MEAN, ymax = MEAN + 3*SE, ymin = MEAN - 3*SE), data = metrics_cv_filt_summary,
                  color = 'black', size = 0.25) +
  # geom_boxplot(aes(x = MODEL, y = VALUE), data = metrics_cv_filt, fill = 'grey50') +
  facet_wrap(METRIC~GROUP, scales = 'free') +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25),
        strip.text.x = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        strip.background = element_blank(),
        legend.position = 'none') +
  scale_x_discrete(name = '')

########################################################################################################################
########################################################################################################################

metrics_roc =
  read_tsv('~/WTCHG/Projects/eQTLseq/results/geuvadis/metrics_roc_alt3.txt') %>%
  mutate(MODEL = ifelse(MODEL == 'Normal', TRANS, MODEL),
         MODEL = stringr::str_replace(MODEL, 'boxcox', 'bcox'),
         MODEL = stringr::str_replace(MODEL, 'NBinomial', 'nbin'),
         MODEL = stringr::str_replace(MODEL, 'Binomial', 'bin'),
         MODEL = stringr::str_replace(MODEL, 'Poisson', 'pois'),
         MODEL = factor(MODEL, levels = c('nbin', 'bin', 'pois', 'log', 'logit',
                                          'arcsin', 'bcox', 'blom', 'voom', 'vst'), ordered = T)) %>%
  select(-TRANS) %>%
  gather(METRIC, VALUE, -MODEL, -NSAMPLES, -GROUP, convert = T)

metrics_roc %>%
  filter(NSAMPLES == 339, METRIC %in% c('MCC', 'FDR'), GROUP %in% c('miRNAs', 'protein_coding')) %>%
  ggplot() +
    geom_jitter(aes(x = MODEL, y = VALUE), width = 0.2, size = 0.2, color = 'gray') +
    geom_boxplot(aes(x = MODEL, y = VALUE), width = 0.2, outlier.shape = NA) +
    scale_y_continuous(trans = 'identity') +
    facet_grid(METRIC~GROUP, scales = 'free_y')

########################################################################################################################
########################################################################################################################

hits = local({
  in_dir = '~/WTCHG/Projects/eQTLseq/results/geuvadis/hits/'
  mdls = list(
    'nbin' = 'NBinomial.none',
    'bin' = 'Binomial.none',
    'pois' = 'Poisson.none',
    'log' = 'Normal.log',
    'logit' = 'Normal.logit',
    'arcsin' = 'Normal.arcsin',
    'blom' = 'Normal.blom',
    'bcox' = 'Normal.boxcox',
    'vst' = 'Normal.vst',
    'voom' = 'Normal.voom'
  )

  miRNAs = plyr::llply(mdls, function(mdl) {
    fin = paste(in_dir, 'TF.common.HIGH.miRNAs.452.', mdl, '.hits.txt', sep = '')
    read_delim(fin, col_names = F, delim = ' ')
  })

  protein_coding = plyr::llply(mdls, function(mdl) {
    fin = paste(in_dir, 'TF.common.HIGH.protein_coding.452.', mdl, '.hits.txt', sep = '')
    read_delim(fin, col_names = F, delim = ' ')
  })

  variants = plyr::llply(names(mdls), function(mdl) {
    miRNAs = colSums(miRNAs[[mdl]]) > 0
    protein_coding = colSums(protein_coding[[mdl]]) > 0
    data_frame(miRNAs = miRNAs, protein_coding = protein_coding) %>%
      mutate(model = mdl) %>%
      rownames_to_column('variant')
  }) %>%
    bind_rows() %>%
    gather(group, value, miRNAs, protein_coding) %>%
    mutate(variant = as.integer(variant))

  # genes = plyr::llply(raw, function(hits) {
  #   rowSums(hits) > 0
  # }) %>%
  #   as_data_frame() %>%
  #   rownames_to_column('gene') %>%
  #   gather(model, value, -gene)

  ##
  list(miRNAs = miRNAs, protein_coding = protein_coding, variants = variants)
})

hits$variants %>%
  filter(value) %>%
  mutate(model = factor(model, levels = c('nbin', 'bin', 'pois', 'log', 'logit', 'arcsin', 'bcox', 'blom', 'voom', 'vst'))) %>%
  ggplot() +
  geom_bar(aes(x = model, fill = group), position = position_dodge()) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        strip.background = element_blank(),
        legend.position = c(0.5, 0.7),
        legend.title = element_blank()) +
  scale_x_discrete(name = '') +
  scale_y_continuous(name = '# eQTLs', breaks = seq(0, 40, by = 2))

hits$variants %>%
  filter(value) %>%
  group_by(variant) %>%
  summarize(n = length(unique(model))) %>%
  ungroup() %>%
  ggplot() +
  geom_bar(aes(x = as.factor(variant), y = n), stat = 'identity', position = position_dodge()) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25),
    # axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        strip.background = element_blank(),
        legend.position = c(0.5, 0.7),
        legend.title = element_blank()) +
  scale_x_discrete(name = 'eQTL ID') +
  scale_y_continuous(name = '# models', breaks = seq(0, 10, by = 1))

# hits$variants %>%
#   filter(value) %>%
#   ggplot() +
#   geom_bar(aes(x = variant, fill = group), position = position_dodge()) +
#   theme_bw() +
#   theme(strip.background = element_blank(),
#         legend.position = c(0.5, 0.7),
#         legend.title = element_blank()) +
#   scale_x_continuous(name = 'variant ID', breaks = seq(1, 40, by = 2), limits = c(1, 39)) +
#   scale_y_continuous(name = '# supporting models', breaks = seq(0, 10, by = 2))

eQTLs = hits$variants %>%
  filter(value) %>%
  group_by(variant, group) %>%
  summarize(models = stringr::str_c(model, collapse = ',')) %>%
  ungroup() %>%
  spread(group, models)

tmp = read_tsv('~/WTCHG/Projects/eQTLseq/results/geuvadis/variants.TF.common.HIGH.txt',
               col_names = F) %>%
  rownames_to_column('ID') %>% mutate(ID = as.integer(ID))

out = left_join(eQTLs, tmp, by = c('variant' = 'ID')) %>%
  filter(variant %in% c(3,4,5,22,23))
