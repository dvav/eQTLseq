##
compute_p = function(cnts) {
  ngenes = nrow(cnts)
  libsizes = colSums(cnts)
  sweep(cnts + 1, 2, libsizes + ngenes, "/")
}

##
arcsin = function(cnts) {
  compute_p(cnts) %>% sqrt() %>% asin()
}

##
logit = function(cnts) {
  p = compute_p(cnts)
  log(p / (1 - p))
}

blom = function(cnts) {
  c = 3/8
  n = nrow(cnts)
  apply(cnts, 2, function(x) {
    r = rank(x)
    p = (r - c) / (n - 2 * c + 1)
    qnorm(p)
  })
}

bcox = function(cnts) {
  apply(cnts, 2, function(x) {
    lam = car::powerTransform(x, family = 'bcPower')$roundlam
    car::bcPower(x, lam)
  })
}

##
COUNTS = list(
  miRNAs = read_tsv('~/Data/Geuvadis/counts/counts_miRNAs.txt') %>% select(-GENE),
  protein_coding = read_tsv('~/Data/Geuvadis/counts/counts_genes_protein_coding.txt') %>% select(-GENEID, -BIOTYPE, -GENE)
)

##
FILTERED = with(COUNTS, list(
  miRNAs = miRNAs %>% filter(rowMeans(.) > 10),
  protein_coding = protein_coding %>% filter(rowMeans(.) > 10)
))

##
MODELS = list()
MODELS$log = with(FILTERED, list(miRNAs = log(miRNAs + 1), protein_coding = log(protein_coding + 1)))
MODELS$blom = with(FILTERED, list(miRNAs = blom(miRNAs), protein_coding = blom(protein_coding)))
MODELS$bcox = with(FILTERED, list(miRNAs = bcox(miRNAs + 1), protein_coding = bcox(protein_coding + 1)))
MODELS$arcsin = with(FILTERED, list(miRNAs = arcsin(miRNAs), protein_coding = arcsin(protein_coding)))
MODELS$logit = with(FILTERED, list(miRNAs = logit(miRNAs), protein_coding = logit(protein_coding)))
MODELS$voom = with(FILTERED, list(miRNAs = limma::voom(miRNAs)[[1]],
                                  protein_coding = limma::voom(protein_coding)[[1]]))
MODELS$vst = with(FILTERED, list(miRNAs = DESeq2::varianceStabilizingTransformation(as.matrix(miRNAs)),
                                 protein_coding = DESeq2::varianceStabilizingTransformation(as.matrix(protein_coding))))


##
STATS = plyr::llply(names(MODELS), function(tag) {
  mdl = MODELS[[tag]]
  bind_rows(
    data_frame(mu = apply(mdl$miRNAs, 1, mean), sd = apply(mdl$miRNAs, 1, sd), group = 'miRNAs'),
    data_frame(mu = apply(mdl$protein_coding, 1, mean), sd = apply(mdl$protein_coding, 1, sd), group = 'protein_coding')
  ) %>% mutate(model = tag)
}) %>%
  bind_rows() %>%
  group_by(group, model) %>%
  mutate(rank = rank(mu)) %>%
  ungroup()

STATS %>%
  ggplot(aes(rank, sd^2, color = model)) +
  # geom_point(size = 0.01) +
  geom_abline(linetype = 'dashed') +
  geom_smooth(method = 'loess', se = F) +
  scale_x_continuous(trans = 'identity') +
  scale_y_continuous(trans = 'identity') +
  facet_wrap(~group, scales = 'free')
