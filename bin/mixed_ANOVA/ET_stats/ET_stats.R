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
  data[ ,sp_gr_in := interaction(condition, group)]
  
  # Mixed ANOVA model
  ezANOVA_res <- ezANOVA(data = data, 
                         dv = .(value), 
                         wid = .(subj_id), 
                         within = .(condition), 
                         between = .(group), 
                         detailed = TRUE)
  
  # Fit the same model with lme
  lme_anova <- lme(value ~ condition*group, 
                   random = ~1|subj_id/condition, 
                   data = data, 
                   method = "ML")
  
  # Fit the same model with lme for post-hoc tests
  lme_ph <- lme(value ~ sp_gr_in, 
                random = ~1|subj_id/condition, 
                data = data, 
                method = "ML")
  
  # Get specified post-hocs with single-step correction
  post_hocs <- glht(lme_ph, linfct = mcp(sp_gr_in = ph_contrasts))
  ph_summary <- summary(post_hocs, test = adjusted(type = 'single-step'))
  
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
data <- melt(id.vars = c('condition', 'group', 'subj_id'),
             measure.vars = colnames(data)[-match(c('condition', 'group', 'subj_id'), colnames(data))],
             variable.name = 'measure', data = data)

# Loop for measures
for (meas in p$meas_list){
  # Perform mixed ANOVA
  mixed_ANOVA_res <- mixed_ANOVA_with_ph(data, meas, p$ph_contrasts)
  
  # Save stats (visualization is done in Python)
  ezANOVA_res = mixed_ANOVA_res$ezANOVA$ANOVA
  lme_anova = anova(mixed_ANOVA_res$lme_anova)
  ph_summary = subset(tidy(mixed_ANOVA_res$ph_summary), select = -rhs)
  save(ezANOVA_res, lme_anova, ph_summary, file = sprintf("%s/ET_ANOVA_%s.RData", p$stats_dir, meas))
}
