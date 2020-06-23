library(data.table, warn.conflicts = FALSE)
library(broom)
library(yaml)
library(ez)
library(nlme)
library(multcomp, warn.conflicts = FALSE)

# Get params
p <- read_yaml('params.yaml')
dir.create(p$stats_dir, recursive = TRUE, showWarnings = FALSE)

# Function for mixed ANOVA with post-hocs
mixed_ANOVA_with_ph <- function(data, measure_name, ph_contrasts){
  data <- data[measure == measure_name]
  data[ ,sp_gr_in := interaction(factor_1, factor_2)]
  
  # Mixed ANOVA model
  ezANOVA_res <- ezANOVA(data = data,
                         dv = .(value),
                         wid = .(subj_id),
                         within = .(factor_1,factor_2),
                         detailed = TRUE)
  
  # Fit the same model with lme
  lme_anova <- lme(value ~ factor_1*factor_2,
                   random = ~1|subj_id/factor_1/factor_2,
                   data = data, 
                   method = "ML")
  
  # Fit the same model with lme for post-hoc tests
  lme_ph <- lme(value ~ sp_gr_in, 
                random = ~1|subj_id/factor_1/factor_2,
                data = data,
                method = "ML")
  
  # Get specified post-hocs with single-step correction
  post_hocs <- glht(lme_ph, linfct = mcp(sp_gr_in = 'Tukey'))
  ph_summary <- summary(post_hocs, test = adjusted(type = 'none'))
  
  # Collect all results in a list
  results_list <- list("ezANOVA" = ezANOVA_res, 
                       "lme_anova" = lme_anova,
                       "lme_ph" = lme_ph,
                       "ph_summary" = ph_summary)
  
  message(paste(measure_name, " Done."))
  return(results_list)
}

# Load data
data <- data.table(read.csv(p$data_path, header = TRUE, sep = ','))

# Long form
data <- melt(id.vars = c('factor_1', 'factor_2', 'subj_id'),
             measure.vars = colnames(data)[-match(c('factor_1', 'factor_2', 'subj_id'), colnames(data))],
             variable.name = 'measure', data = data)

# Loop for measures
for (measure_name in p$meas_list){
  # Perform mixed ANOVA
  mixed_ANOVA_res <- mixed_ANOVA_with_ph(data, measure_name, p$ph_contrasts)
  
  # Save stats (visualization is done in Python)
  ezANOVA_res = mixed_ANOVA_res$ezANOVA$ANOVA
  lme_anova = anova(mixed_ANOVA_res$lme_anova)
  ph_summary = subset(tidy(mixed_ANOVA_res$ph_summary), select = -rhs)
  save(ezANOVA_res, lme_anova, ph_summary, file = sprintf("%s/mixed_ANOVA_%s.RData", p$stats_dir, measure_name))
  
  print(mixed_ANOVA_res$ezANOVA)
  print(lme_anova)
  print(mixed_ANOVA_res$ph_summary)
}
