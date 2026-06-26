#tmp_tarball <- "/Users/nickschlitter/Downloads/15495748/germaparl_v2.3.0-rc1.tar.gz"

#devtools::install_github("PolMine/cwbtools")

library(cwbtools)
#corpus_install(tarball = tmp_tarball)

# install polmineR
#install.packages("polmineR")

# check installation
library(polmineR)
corpus("GERMAPARL2")

library(reticulate)
library(dplyr)
library(stringr)
library(purrr)
library(tibble)
library(arrow)

use_virtualenv("r-tf-env", required = TRUE) #Virtuelle Umgebung zur Ausführung der Python Surprisal Berechnung
source_python("gpt2_surprisal_optimized.py")



#bundestag_parties <- s_attributes(corpus("GERMAPARL2"), "speaker_party") |>
#  setdiff(c("", "fraktionslos", "parteilos"))

#Da die Beta-Version des Korpus noch nicht entsprechend funktioniert, müssen die Daten manuell extrahiert werden.
#"AfD", "CDU", "SPD" "CSU", "FDP", "GRUENE", "DIE LINKE"


party_speeches <- corpus("GERMAPARL2") |>
  subset(p_type == "speech") |>
  subset(speaker_party == "AfD") |> #Hier die Partei Anpassen
  subset(protocol_year >= 2017)


speeches <- as.speeches(party_speeches, s_attribute_name = "speaker_name")
#Speeches Objekt zu clean Text
readable_party_speeches <- polmineR::as.VCorpus(speeches)
readable_party_speeches$content[[1]]$meta


final_speeches <- readable_party_speeches$content |>
  imap_dfr(~ {
    key <- .y
    doc <- .x

    m <- str_match(key, "(.+?)_(\\d{4}-\\d{2}-\\d{2})")
    if (is.na(m[1,1])) return(NULL)

    tibble(
      name = m[1,2],
      date = as.Date(m[1,3]),
      text = doc$content
    )
  })

final_speeches$surprisal <- 0.00


for (i in 1:NROW(final_speeches)){
  final_speeches$surprisal[i] <- py$compute_surprisal(final_speeches$text[i])
  print(i)
}

write.csv(final_speeches, "afd_speeches_surprisal.csv")
