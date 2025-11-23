# ===============================================================
# R SCRIPT KLASIFIKASI LULC L4 (Revisi - Batas Sampel)
# Training: Data 2024
# Aplikasi: Time Series Data
# ===============================================================

# 1. Muat library
library(terra)
library(sf)
library(Boruta)
library(randomForest)
library(dplyr)

# ---------------------------------------------------------------
# 2. Baca data raster tahun 2024 
# ---------------------------------------------------------------
S2SEV_2024 <- rast("S2SEV24.tif")
names(S2SEV_2024) <- make.names(names(S2SEV_2024), unique = TRUE)

# ---------------------------------------------------------------
# 3. Baca data training shapefile (2024)
# Target klasifikasi: Id
# ---------------------------------------------------------------
train_shp <- st_read("ST_seascape_training.shp")

# Ekstrak training sample (berbasis poligon)
train_vals <- terra::extract(S2SEV_2024, vect(train_shp), df = TRUE)

# Tambahkan kolom Id dari shapefile sesuai index ID
train_vals$Id <- train_shp$Id[train_vals$ID]

# Bersihkan kolom tidak relevan
train_vals$ID <- NULL
train_vals$class <- NULL
train_vals$geometry <- NULL

# Hilangkan NA
train_data <- na.omit(train_vals)

# Pastikan faktor
train_data$Id <- as.factor(train_data$Id)

cat("Jumlah sampel per kelas (sebelum dibatasi):\n")
print(table(train_data$Id))

# ---------------------------------------------------------------
# 4. Feature Selection dengan Boruta
# ---------------------------------------------------------------
set.seed(123)
boruta_result <- Boruta(Id ~ ., data = train_data,
                        doTrace = 1, ntree = 500) 

final_boruta <- TentativeRoughFix(boruta_result)
selected_vars <- getSelectedAttributes(final_boruta, withTentative = FALSE)

cat("Variabel terpilih hasil Boruta:\n")
print(selected_vars)

# Simpan hasil Boruta
boruta_df <- attStats(final_boruta)
write.csv(boruta_df, "Boruta_Result.csv", row.names = TRUE)

# ---------------------------------------------------------------
# 5. Sampling Training 
# ---------------------------------------------------------------
library(dplyr)

set.seed(123)
train_sample <- train_data %>%
  dplyr::select(all_of(c("Id", selected_vars))) %>%
  group_by(Id) %>%
  group_modify(~ {
    n_class <- nrow(.x)
    if (n_class > 1000) dplyr::slice_sample(.x, n = 1000) else .x
  }) %>%
  ungroup()

cat("Jumlah sampel per kelas (setelah dibatasi):\n")
print(table(train_sample$Id))

# Simpan data training final
write.csv(train_sample, "TrainingData_Final_RF.csv", row.names = FALSE)

# ---------------------------------------------------------------
# 6. Model Random Forest
# ---------------------------------------------------------------
set.seed(123)
rf_model <- randomForest(
  x = train_sample[, selected_vars], 
  y = train_sample$Id, 
  ntree = 500
)

# ---------------------------------------------------------------
# 6a. Evaluasi OOB Error
# ---------------------------------------------------------------
print(rf_model)

oob_error <- rf_model$err.rate[rf_model$ntree, "OOB"]
cat("OOB Error keseluruhan:", round(oob_error * 100, 2), "%\n")

conf_mat <- rf_model$confusion
write.csv(conf_mat, "OOB_ConfusionMatrix.csv")

# ---------------------------------------------------------------
# ==========================================
# PENGATURAN OPSI MEMORI TERRA
# ==========================================
terraOptions(
  memfrac = 0.75,                 # gunakan 60% RAM
  tempdir = "/Volumes/T7/Sulteng_Benthic/S2SEV"        # folder sementara dengan ruang besar
)

# ==========================================
# PREDIKSI 2024
# ==========================================
terraOptions(
  memfrac = 0.75,
  tempdir = "/Volumes/T7/Sulteng_Benthic/S2SEV"
)

pred_2024 <- predict(
  S2SEV_2024[[selected_vars]],
  rf_model,
  fun = function(model, data, ...) {
    predict(model, newdata = as.data.frame(data), type = "response")
  },
  na.rm = TRUE,
  filename = "ST_ben_24.tif",
  overwrite = TRUE,
  wopt = list(datatype="INT1U", gdal=c("COMPRESS=LZW"))
)

# ==========================================
# PREDIKSI 2019
# ==========================================
S2SEV_2019 <- rast("S2SEV19.tif")
names(S2SEV_2019) <- make.names(names(S2SEV_2024), unique = TRUE)

pred_2019 <- predict(
  S2SEV_2019[[selected_vars]],
  rf_model,
  fun = function(model, data, ...) {
    predict(model, newdata = as.data.frame(data), type = "response")
  },
  na.rm = TRUE,
  filename = "ST_ben_19.tif",
  overwrite = TRUE,
  wopt = list(datatype="INT1U", gdal=c("COMPRESS=LZW"))
)
