gg_mafs = read_csv('data/mafs.txt', col_names = 'MAF') %>%
  ggplot() +
    geom_histogram(aes(x = MAF), bins = 100, color = 'black', size = 0.25) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))

gg_pars = read_csv('data/pars.txt') %>%
  ggplot() +
    geom_point(aes(x = mu, y = mu + phi * mu^2), size = 0.01) +
    geom_abline(linetype = 'dashed', size = 0.25) +
    scale_x_continuous(trans = 'log10') +
    scale_y_continuous(trans = 'log10') +
    theme_classic() +
    theme(axis.line = element_line(size = 0.25))


gg_folds = read_csv('data/folds.txt', col_names = 'FOLD') %>%
  rownames_to_column('ID') %>%
  mutate(ID = as.numeric(ID)) %>%
  filter(FOLD != 1) %>%
  ggplot() +
  geom_segment(aes(x = ID, xend = ID, y = 1, yend = FOLD), size = 0.25) +
  geom_hline(yintercept = 1, linetype = 'dashed', size = 0.25) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))


gg_beta = read_csv('data/beta.txt', col_names = 'BETA') %>%
  rownames_to_column('ID') %>%
  mutate(ID = as.numeric(ID)) %>%
  filter(BETA != 0) %>%
  ggplot() +
  geom_segment(aes(x = ID, xend = ID, y = 0, yend = BETA), size = 0.25) +
  geom_hline(yintercept = 0, linetype = 'dashed', size = 0.25) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))


gg_cnts = read_tsv('data/montpick_count_table_CEU.txt')[,-1][sample(1e4, 1000, replace = F),sample(60, 10, replace = F)] %>%
  filter(rowMeans(.) > 0) %>%
  gather(Sample, Value, everything()) %>%
  ggplot() +
  geom_histogram(aes(x = Value + 1, y = ..density..), bins = 100, color = 'black', size = 0.25) +
  scale_x_continuous(trans = 'log10') +
  scale_y_continuous(limits = c(0,2)) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))


gg_sim = read_delim('data/cnts.txt', col_names = FALSE, delim = ' ') %>%
  filter(rowMeans(.) > 0) %>%
  gather(Sample, Value, everything()) %>%
  ggplot() +
  geom_histogram(aes(x = Value + 1, y = ..density..), bins = 100, color = 'black', size = 0.25) +
  scale_x_continuous(trans = 'log10') +
  scale_y_continuous(limits = c(0,2)) +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))


plot_grid(gg_mafs, gg_pars, gg_beta, gg_folds, gg_cnts, gg_sim, align = 'hv', ncol = 2, labels = 'auto')
