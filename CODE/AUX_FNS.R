#AUXILIARY FUNCTIONS

#Summary for NA_values & duplicates
duplicates <- function(data, identifier_column) {
  summary <- data %>% 
    summarise(
      na_count = sum(is.na({{ identifier_column }})),
      duplicated_count = sum(duplicated({{ identifier_column }}))
    )
  return(summary)
}




import_gdp_data <- function(file_path) {
  data <- read_excel(file_path) %>%
    select(Trimestres, P.I.B.R.) %>%
    mutate(Trimestres = as.character(Trimestres), P.I.B.R. = as.double(P.I.B.R.)) %>%
    filter(grepl("\\.I$|\\.II$|\\.III$|\\.IV$", Trimestres))
  return(data)
}



create_ts_entry <- function(data, year) {
  data %>%
    mutate(Trimestres = paste0(Trimestres, "_Qrt", gsub(".*\\.(\\w+)$", "\\1", Trimestres), "_", year))
}





