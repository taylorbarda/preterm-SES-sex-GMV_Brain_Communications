# Runs a list of PROCESS (Hayes v4.3) models.
# Called from bc_utils.run_process_jobs():
#   Rscript run_process.R <process.R> <jobs.csv> <data_dir> <out_dir> <boot> <seed>
#
# jobs.csv columns: data, y, x, w, z, cov, model, output_name, complete_cases
#   data  = name of <data_dir>/<data>.csv
#   z     = "" for model 1
#   cov   = covariates separated by ";"

args <- commandArgs(trailingOnly = TRUE)
process_file <- args[1]
jobs_file    <- args[2]
data_dir     <- args[3]
out_dir      <- args[4]
boot         <- as.integer(args[5])
seed         <- as.integer(args[6])

options(width = 200)  # keep PROCESS coefficient tables on one line
invisible(capture.output(source(process_file, chdir = TRUE)))

jobs <- read.csv(jobs_file, stringsAsFactors = FALSE, na.strings = character(0))

datasets <- list()
for (key in unique(jobs$data)) {
  datasets[[key]] <- read.csv(file.path(data_dir, paste0(key, ".csv")), check.names = FALSE)
}

run_job <- function(job) {
  data <- datasets[[job$data]]
  covs <- strsplit(job$cov, ";", fixed = TRUE)[[1]]
  z    <- if (nchar(job$z) > 0) job$z else "xxxxx"

  if (as.logical(job$complete_cases)) {
    needed <- c(job$y, job$x, job$w, covs, if (z != "xxxxx") z)
    data <- data[complete.cases(data[, needed]), needed]
  }

  output_file <- file.path(out_dir, job$output_name)

  status <- tryCatch({
    sink(output_file)
    process(
      data = data, y = job$y, x = job$x, w = job$w, z = z, cov = covs,
      model = job$model, boot = boot, conf = 95, center = FALSE,
      plot = 1, save = 1, seed = seed
    )
    "completed"
  }, error = function(e) {
    paste("failed:", conditionMessage(e))
  }, finally = {
    while (sink.number() > 0) sink()
  })

  data.frame(output_name = job$output_name, n = nrow(data), status = status)
}

log <- do.call(rbind, lapply(seq_len(nrow(jobs)), function(i) run_job(jobs[i, ])))
write.csv(log, file.path(data_dir, "jobs_log.csv"), row.names = FALSE)
