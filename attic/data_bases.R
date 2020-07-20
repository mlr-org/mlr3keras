library(nycflights13)
path = tempfile("flights", fileext = ".sqlite")
flights$row_id = 1:nrow(flights)
con = DBI::dbConnect(RSQLite::SQLite(), path)
tbl = DBI::dbWriteTable(con, "flights", as.data.frame(flights))
DBI::dbDisconnect(con)

# remove in-memory data
rm(flights)


# establish connection
con = DBI::dbConnect(RSQLite::SQLite(), path)

# select the "flights" table, enter dplyr
library("dbplyr")
tbl = tbl(con, "flights")
keep = c("row_id", "year", "month", "day", "hour", "minute", "dep_time",
  "arr_time", "carrier", "flight", "air_time", "distance", "arr_delay")
tbl = select(tbl, keep)
tbl = filter(tbl, !is.na(arr_delay))
tbl = filter(tbl, row_id %% 2 == 0)
tbl = mutate(tbl, carrier = case_when(
    carrier %in% c("OO", "HA", "YV", "F9", "AS", "FL", "VX", "WN") ~ "other",
    TRUE ~ carrier)
)


library("mlr3")
library("mlr3db")
dplyr::show_query(tbl)


db = as_sqlite_backend(tsk("iris")$data())
dplyr::show_query(db$.__enclos_env__$private$.data)
task = TaskClassif$new("iris_db", db, target = "Species")

library(tfdatasets)
library(mlr3misc)



record_spec = sql_record_spec(
  names = task$feature_types$id,
  types = map(task$feature_types$type, function(type) {
      tftypes = switch(type,
        numeric = tf$float64,
        integer = tf$int32
      )
  })
)


stringi::stri_sub(
  dplyr::show_query(task$backend$.__enclos_env__$private$.data),
  1, 5)

dataset = sqlite_dataset(
  "data/mtcars.sqlite3",
  "select * from mtcars",
  record_spec
)

dataset