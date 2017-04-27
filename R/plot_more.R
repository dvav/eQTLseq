gg_state = read_tsv('notebooks/tmp.txt', col_names = 'State') %>%
  add_column(Iter = 1:length(.$State), .before = 1) %>%
  slice(20:200) %>%
  ggplot() +
  geom_line(aes(x = Iter, y = State), size = 0.25, linetype = 'dashed', color = 'black') +
  geom_point(aes(x = Iter, y = State), size = 0.5, shape = 21, color = 'black') +
  scale_x_continuous('iteration') +
  scale_y_continuous('state') +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))

gg_hits = data_frame(x = 600000, y = 0, xend = 600000, yend = 0.83, xx = 600000, yy = 0.69) %>%
  ggplot() +
  geom_hline(yintercept = 0, linetype = 'dashed', size = 0.25) +
  geom_segment(aes(x, y, xend = xend, yend = yend), size = 0.25) +
  geom_point(aes(xx, yy), color = 'black') +
  scale_x_continuous('genes x markers', limits = c(0, 1e6)) +
  scale_y_continuous('effect size') +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))


cowplot::plot_grid(gg_state, gg_hits, align = 'hv', nrow = 2, labels = 'auto')


gg_mat = read_delim('data/1000G_chr7_100K_200K_005pc_5pc.txt', col_names = as.character(1:435), delim = ' ') %>%
  as.matrix() %>%
  {t(.) %*% .} %>%
  reshape2::melt(varnames = c('X', 'Y'), value.name = 'Z') %>%
  mutate(X = as.integer(X), Y = as.integer(Y)) %>%
  ggplot() +
  geom_tile(aes(x = X, y = Y, fill = Z)) +
  coord_equal() +
  theme_classic() +
  theme(legend.position = 'none',
        axis.line = element_line(size = 0.25)) +
  scale_x_continuous('markers', breaks = seq(0, 400, by = 50), expand = c(0,0)) +
  scale_y_continuous('markers', breaks = seq(0, 400, by = 50), expand = c(0,0), trans = 'reverse') +
  scale_fill_gradient(low = 'white', high = 'black')

gg_hits = data_frame(x = c(110280, 157980),
                     y = c(0, 0),
                     xend = c(110280, 157980),
                     yend = c(-0.68458451, 1.115485),
                     xx = c(110280, 157983),
                     yy = c(-0.67696773, 0.32746483)) %>%
  ggplot() +
  geom_hline(yintercept = 0, linetype = 'dashed', size = 0.25) +
  geom_segment(aes(x, y, xend = xend, yend = yend), size = 0.25) +
  geom_point(aes(xx, yy), color = 'black') +
  coord_equal() +
  scale_x_continuous('genes x markers', labels = paste(seq(110, 160, by = 10), 'K', sep = '')) +
  scale_y_continuous('effect size') +
  theme_classic() +
  theme(axis.line = element_line(size = 0.25))

cowplot::plot_grid(gg_mat, gg_hits, align = 'hv', nrow = 1, labels = 'auto')
