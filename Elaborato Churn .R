
################################  LIBRARIES ################################
library(tidyverse)
library(skimr)
library(corrplot)
library(vcd)
library(rpart)  # Load the rpart package
library(caret) # per training-test split
library(rpart)  
library(rpart.plot)
library(coefplot)
library(randomForest)
library(C50) #boost
library(CustomerScoringMetrics)
library(gains)
library(gridExtra)
library(e1071)


################################  SETTINGS ################################
getwd() 
#setwd("/Users/gpoggi/Desktop/Courses/mercatorum/Elaborato Churn")
path <- 'Churn_Banking_Modeling_Anonym.csv'

#scientific notaiton
# Set the scipen option to a high value
options(scipen = 999)


hex_base = '#58bc84'
#  palettes 
palette_ld <- colorRampPalette(c("white", hex_base))(n = 5)
palette_dl <- colorRampPalette(c(hex_base, "white"))(n = 5)  # Adjust the number of shades as needed
contrasting_palette <- c('#4e024f','#811a85', '#655986', '#8be488' , '#58bc84')
palette_f <- colorRampPalette(c('#4e024f', 'white', '#58bc84'))

################################  LOAD DATA ################################ 

#df <- read.table(path, header = TRUE, sep = ',', encoding = ('latin1') )
df <- read_csv(path, locale = locale(encoding = 'Latin1'))

#####################################################################
#################    BUSINESS UNDERSTANDING   #######################
#####################################################################

# Siamo all'interno di un progetto che ha, come obiettivo finale, la costruzione di un modello di Data Mining
# per la previsione dell'abbandono del cliente. 
# var_target indica se il cliente ha abbandonato (churn) dopo il periodo di osservazione.

#####################################################################
#################    DATA UNDERSTANDING   ##########################$
#####################################################################

##################################### EDA #########################

# nomi colonne to lower case
colnames(df) <- tolower(colnames(df))

#prime righe
head(df)%>% view()

#salvare nomi colonne in variabile
cols <- colnames(df)

# esplorazione rapida df
summary_df <- skim(df)
View(summary_df)
### osservazioni 
# shape 41 x 377369
# # molte variabili ind_ sono indicatori categorici
# molte variabili hanno una complete rate molto bassa. Da valutare se eliminarle subito o considerare che possano essere degli 0
# sesso ha 3 valori unici, da controllare (potrebbe essere diverse o potrebbe essere un typo)
# professione ha 17 valori unici, possibile ridurre?
# variabile target unbalanced
# varibili utili: 
# ind_variazione_accredito_stipendio, ind_contatto_call_center, ind_membro_programma_loyalty, ind_giroconto,
# ind_prestito, ind_accredito_stipendio, ind_multi_account, ind_conto_online, 
# altri ind_ sembrano non variare (creare selezionatore di variazione?) 
# ind_disattivazione_rid,ind_rifiuto_prestiti, ind_rifiuto_carte, ind_trasferimento_titoli, 
# ind_richiesta_info_chiusura_conto, 


#selezione variabili con complete rate superiore alla soglia
threshold <- 0.8  # Soglia desiderata

# Filtra le colonne con complete_rate superiore alla soglia
selected_columns <- summary_df$skim_variable[summary_df$complete_rate > threshold]

# Stampa la lista delle colonne selezionate
print(selected_columns)
print(length(selected_columns))

# da 40 a 26 variabili

#backup dataframe se voglio tornare a df originario dopo
df_copy <- data.frame(df)
#ridurre dataframe a colonne selezionate in base a complete rate
# in una prima fase e senza ulteriori info dal business, non assumo di poter imputare un numero superiore al 20% dei dati
# con successive riunioni é possibile intuire se alcune imputazioni sono facili (come missing = 0 per numeriche)
# e possibili anche per le variabili scartate ora

df <- subset(df, select = selected_columns)

head(df)%>% view()


#controllare tipo delle colonne
str(df) 

#colonne numeriche
num_cols <- colnames(df[,sapply(df,is.numeric)])
num_cols <- setdiff(num_cols, "id_cliente") #rimuovi id_cliente

print(num_cols)

#colonne non numeriche 
cat_cols <- colnames(df[,!sapply(df,is.numeric)])
print(cat_cols)

#colonne che sono indicatori 
ind_cols <- grep("^ind_", colnames(df), value = TRUE)


#imputazione NA colonne numeriche


#etá non possiamo imputare a 0 ma media

# Imputiamo eta' media ai missing data
df <- df %>% 
  mutate(eta = case_when(
    is.na(eta) == TRUE ~ round(mean(eta, na.rm = TRUE), digits = 0),
    TRUE ~ eta)
  )

## questo opera su tutte tranne "eta" 
df <- df %>% 
  mutate(across(where(is.numeric) & contains(match = c("amm_", "cont_")), ~ replace_na(data = .x, replace = 0)))
skim(df)%>%View()

## caratteri vuoti per sesso, profilo_mifid, professione
df <- df %>% 
  mutate(across(.cols = c('sesso','profilo_mifid','professione'), ~ replace_na(data = .x, replace = 'ND')))


#skim(df)%>%View()

# distribuzione var target

# bar plot var_target

df %>%
ggplot(aes(x = var_target, fill= var_target)) +
  geom_bar()+
  scale_fill_manual(values= contrasting_palette[c(1,5)])+
  ggtitle('Target Variable Distribution')


# bar_plot variabili categoriche

for (i in cat_cols) {
  p<- df %>%
    ggplot(aes(x = .data[[i]], fill= .data[[i]])) +
    geom_bar()+
    ggtitle(paste(i, 'distribution'))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  print(p)
}

print(setdiff(num_cols, ind_cols))

#hist numerical variables
for (i in setdiff(num_cols, ind_cols)) {
  p<- df %>%
    ggplot(aes(x = .data[[i]], fill= .data[[i]])) +
    geom_histogram(color=hex_base, fill=hex_base)+
    ggtitle(paste(i, 'distribution'))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  print(p)
}

#hist numerical variables (LOG)
for (i in setdiff(num_cols, ind_cols)) {
  p<- df %>%
    ggplot(aes(x = .data[[i]], fill= .data[[i]])) +
    geom_histogram(color=hex_base, fill=hex_base)+
    ggtitle(paste(i, 'distribution Log'))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))+
    scale_x_log10()  # Applica scala logaritmica all'asse x
  print(p)
}

#plot ind_cols variables
for (i in ind_cols) {
  p<- df %>%
    ggplot(aes(x = .data[[i]], fill= .data[[i]])) +
    geom_bar(fill=hex_base)+
    ggtitle(paste(i, 'distribution'))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  print(p)
}


#>
# sesso + professione 
ggplot(data = df, aes(x = var_target)) +
  geom_bar(color = hex_base, fill = hex_base) +
  facet_wrap(~ professione +sesso )

#distrib var target per variabile 
# Costruzione della tabella di contingenza
contingency_table <- table(df$sesso, df$var_target)
relative_contingency_table <- prop.table(contingency_table, margin=2) # relativa a colonne

# Stampa della tabella di contingenza
print(contingency_table)
print(relative_contingency_table) #relativa

barplot(relative_contingency_table, col = contrasting_palette, legend=TRUE)


# Calcolo dell'indice di Cramer
cramer_index <- assocstats(contingency_table)$cramer

# Stampa dell'indice di Cramer
print(cramer_index)


for (i in cat_cols) {
  contingency_table <- table(df[[i]], df$var_target)
  relative_contingency_table <- round(prop.table(contingency_table, margin = 2), 2)
  
  # Stampa della tabella di contingenza
  print(contingency_table)
  print(relative_contingency_table)
  
  par(mar = c(5, 4, 4, 11)) #bottom, left, top, right
  
  # Create the bar plot and store the barplot values
  bar_graph <- barplot(relative_contingency_table, col = contrasting_palette, main = i)
  
  # posizione legend
  legend_x <- max(bar_graph) + 0.5
  legend_y <- max(relative_contingency_table) + 0.5
  
  # legend fuori dalle bar
  legend(legend_x, legend_y, legend = unique(df[[i]]), fill = contrasting_palette, bty = "n", xpd = TRUE)
  
  # Calcolo dell'indice di Cramer
  cramer_index <- assocstats(contingency_table)$cramer
  
  # Stampa dell'indice di Cramer
  print('cramer index')
  print(cramer_index)
}



#correlazione variabili numeriche


corr_matrix <- cor(df %>% select(num_cols))
corrplot(corr_matrix, type = "upper", tl.col = 'black', cl.ratio = 0.1, col = palette_f(10))


#####################################################################
#################   DATA PREPARATION          #######################
#####################################################################

##################################### DATA CLEANING #################

colnames(df)
#cols_to_drop = c( 'id_cliente' # scartare id_cliente)

#df <- df[, !(names(df) %in% cols_to_drop)]
colnames(df)

# to factor

# Mutate ind_cols in factor ##
df <- df %>% 
  mutate(across(all_of(ind_cols), ~ as_factor(.x))) 
#mutate cat cols in factor ##
df <- df %>% 
  mutate(across(all_of(cat_cols), ~ as_factor(.x)))

#pulire  etá (scartare <18 , >85) - Proposta al business

nrow(df[df$eta < 18, ]) #85
nrow(df[df$eta > 85, ]) #1910

# Rimuoviamo clienti con etá fuori dal range

df <- df %>% 
  filter(eta <= 85)
df <- df %>% 
  filter(eta > 18)

#####################################################################
############################   MODELING      #######################
####################################################################

###############################     TRAIN-TEST SPLIT ##############


# Train test split
set.seed(42)  # Seed 


intrain <- createDataPartition(df$var_target, p = 0.8, list = FALSE)
#print(intrain)
training <- df[intrain,]
testing <- df[-intrain,]

###################### [TRAINING]   TARGET UNBALANCE ##############
table(training$var_target)  # distribution non bilanciata

# Undersampling
training_balanced <- training %>%
  group_by(var_target) %>%
  sample_n(min(table(training$var_target))) %>%
  ungroup()

table(training_balanced$var_target)  


colnames(training_balanced)
###############################     Define a function to evaluate ##############

eval_model <- function(model_name, test_data, predictions) { # create a function with the name my_function
  # Misurare performance 
  
  confusion_matrix <- confusionMatrix(predictions, testing$var_target)
  recall <- round(confusion_matrix$byClass["Recall"],2)
  precision <- round(confusion_matrix$byClass["Precision"],2)
  f1_score <- round(confusion_matrix$byClass["F1"],2)
  bal_accuracy <- round(confusion_matrix$byClass["Balanced Accuracy"],2)
  #print(confusion_matrix)
  
  # dataframe di risultati
  results_df <- data.frame(model = model_name, recall = recall, precision=precision, f1_score=f1_score, bal_acc=bal_accuracy)
  rownames(results_df) <- NULL #remove index
  return (results_df)
}

#####################################  BASELINE MODEL ##############
#testing$sesso <- factor(testing$sesso, levels = levels(training$sesso))

# Naive Bayes classifier
model_naive <- naiveBayes(var_target ~ . - id_cliente, data = training_balanced)

#

predictions_naive <- predict(model_naive, newdata = testing, type = "class")

results_df <- eval_model('naive', testing, predictions_naive)

predictions_naive_perc <- predict(object = model_naive, 
                                 newdata = testing %>% select(-var_target), type='raw')[, "si"]




# gain curve for naive model
gain_curve_naive <- gains(ifelse(testing$var_target == "si", 1, 0), 
                         as.numeric(factor(predictions_naive_perc)), 
                         optimal = TRUE, percents = TRUE, groups=50)
gain_curve_naive <- bind_cols(c(0,gain_curve_naive$depth), c(0,gain_curve_naive$cume.pct.of.total*100)) %>% 
  rename(depth = ...1,
         cume.pct.of.total = ...2)




# Modello baseline (tree)
tree_model_baseline <- rpart(var_target ~ . - id_cliente, data = training_balanced)

#Valutazione e salva in dataframe
predictions_baseline_tree <- predict(tree_model_baseline, newdata = testing, type = "class")

results_df <- rbind(eval_model('tree_model_baseline', testing, predictions_baseline_tree),results_df)

# Visualize the decision tree
rpart.plot(tree_model_baseline)

predictions_tree_perc <- predict(object = tree_model_baseline, 
                               newdata = testing %>% select(-var_target), type='prob')[, "si"]

# gain curve tree model
gain_curve_tree <- gains(ifelse(testing$var_target == "si", 1, 0), 
                       as.numeric(factor(predictions_tree_perc)), 
                       optimal = TRUE, percents = TRUE, groups=50)
gain_curve_tree <- bind_cols(c(0,gain_curve_tree$depth), c(0,gain_curve_tree$cume.pct.of.total*100)) %>% 
  rename(depth = ...1,
         cume.pct.of.total = ...2)


# Test della Random Forest

rf_model_baseline <- randomForest(var_target ~ .- id_cliente, data = training_balanced, ntree = 500)
#Valutazione e salva in dataframe
predictions_baseline_RF <- predict(rf_model_baseline, newdata = testing)
results_df <- rbind(eval_model('RF_baseline', testing,predictions_baseline_RF),results_df)

predictions_rf_perc <- predict(object = rf_model_baseline, 
                                newdata = testing %>% select(-var_target), type='prob')[, "si"]

varImpPlot(rf_model_baseline,bg = hex_base)

# gain curve 
gain_curve_rf <- gains(ifelse(testing$var_target == "si", 1, 0), 
                        as.numeric(factor(predictions_rf_perc)), 
                        optimal = TRUE, percents = TRUE, groups=50)
gain_curve_rf <- bind_cols(c(0,gain_curve_rf$depth), c(0,gain_curve_rf$cume.pct.of.total*100)) %>% 
  rename(depth = ...1,
         cume.pct.of.total = ...2)


#####################################  FEATURE ENGINEERING (+ go back to data cleaning/preparation) #########
colnames(training_balanced)


# new feature: anni con la banca dat_apertura_primo_conto
df$anni_fedelta <- c(max(df$dat_apertura_primo_conto)-df$dat_apertura_primo_conto)

# new feature: variazione amm_liquidita_attuale e amm_liquiditá_attuale_6m

mean(df$amm_liquidita_attuale)
mean(df$amm_liquidità_attuale_6m)
# le due misure sembrano essere la stessa (riunione col business per conferma )

#idea: se la liquiditá corrente diminuisce rispetto alla media dei 6 mesi, i clienti stanno pensando a chiudere
df$amm_liquidita_delta <- c(df$amm_liquidita_attuale-df$amm_liquidità_attuale_6m)

#new feature: multipli indicatori negativi

df$ind_trasferimento_cash_titoli <- ifelse(df$amm_liquidita_delta < 0 & df$ind_trasferimento_titoli == 1, 1, 0)
df$ind_contatto_chiusura <- ifelse(df$ind_richiesta_info_chiusura_conto == 1 & df$ind_contatto_call_center == 1, 1, 0)
df$ind_rifiuto <- ifelse(df$ind_rifiuto_prestiti == 1 & df$ind_rifiuto_carte == 1, 1, 0)

colnames(df)
#new feature: cluster basato sugli indicatori
#------ Elbow method 


X = df[ind_cols]
k_values <- 1:10  # Range of K values to consider

# Calculate tot.withinss for each K value
tot_withinss <- sapply(k_values, function(k) kmeans(X, centers = k, nstart = 10)$tot.withinss)

# Find the best K using the "elbow method"
best_k <- which.min(tot_withinss)

# Plot the K values and tot.withinss
ggplot() +
  geom_line(aes(x = k_values, y = tot_withinss), color = "blue") +
  geom_point(aes(x = k_values, y = tot_withinss), color = "red") +
  labs(x = "K value", y = "Total Within-Cluster Sum of Squares") +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "green") +
  ggtitle("Elbow Method: ind_ ")



# K-means clustering model

X = df[ind_cols]
k = 4  # Specify the number of clusters

kmeans_model <- kmeans(X, centers = k, nstart = 10)
#df$ind_cluster <- kmeans_model$cluster

#new feature: cluster_2

# K-means clustering model
# anni_fedelta, professione, profilo_mifid, eta, categoria_cliente

# One hot encoding delle variabili fattore
df_dummies <- model.matrix(~., data = df %>% select(profilo_mifid, professione, categoria_cliente))

# combinazione con altre numeriche
df_dummies <- cbind(df %>% select(anni_fedelta, eta), df_dummies)
#------ Elbow method 


X = df_dummies
k_values <- 1:10  # Range of K values to consider

# Calculate tot.withinss for each K value
tot_withinss <- sapply(k_values, function(k) kmeans(X, centers = k, nstart = 10)$tot.withinss)


# Plot the K values and tot.withinss
ggplot() +
  geom_line(aes(x = k_values, y = tot_withinss), color = "blue") +
  geom_point(aes(x = k_values, y = tot_withinss), color = "red") +
  labs(x = "K value", y = "Total Within-Cluster Sum of Squares") +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "green") +
  ggtitle("Elbow Method: Profili")


#con elbow method: 5 
#uso il dataframe dummies per un clustering basato su queste variabili e creare una nuova variabile 
X = df_dummies
k = 5  # Numero cluster selezionato con Elbow method

kmeans_model <- kmeans(X, centers = k, nstart = 10)
df$cluster_2 <- kmeans_model$cluster



#------ Elbow method (qua: dimostrato su un k means che unisce il clustering per indicatori e su invece i profili cliente)

df_all_cluster = cbind(df %>% select(all_of(ind_cols)), df_dummies)
X = df_all_cluster
k_values <- 1:10  # Range of K values to consider

# Calculate tot.withinss for each K value
tot_withinss <- sapply(k_values, function(k) kmeans(X, centers = k, nstart = 10)$tot.withinss)

# Find the best K using the "elbow method"
best_k <- which.min(tot_withinss)

# Plot the K values and tot.withinss
ggplot() +
  geom_line(aes(x = k_values, y = tot_withinss), color = "blue") +
  geom_point(aes(x = k_values, y = tot_withinss), color = "red") +
  labs(x = "K value", y = "Total Within-Cluster Sum of Squares") +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "green") +
  ggtitle("Elbow Method: Ind_ e Profili")


#con elbow method: 5 


#clustering unico, basato su tutte le feature interessanti

#uso il dataframe dummies per un clustering basato su queste variabili e creare una nuova variabile 
X = df_all_cluster
k = 5  # Specify the number of clusters

kmeans_model <- kmeans(X, centers = k, nstart = 10)
df$cluster <- kmeans_model$cluster


#====== distrubizione dei cluster var_target

for (i in c("cluster", "cluster_2")){

  contingency_table <- table(df[[i]], df$var_target)
  relative_contingency_table <- round(prop.table(contingency_table, margin = 2), 2)
  
  # Stampa della tabella di contingenza
  print(contingency_table)
  print(relative_contingency_table)
  
  par(mar = c(5, 4, 4, 11)) #bottom, left, top, right
  
  # Create the bar plot and store the barplot values
  bar_graph <- barplot(relative_contingency_table, col = contrasting_palette, main = i)
  
  # posizione legend
  legend_x <- max(bar_graph) + 0.5
  legend_y <- max(relative_contingency_table) + 0.5
  
  # legend fuori dalle bar
  legend(legend_x, legend_y, legend = unique(df[[i]]), fill = contrasting_palette, bty = "n", xpd = TRUE)
  
  }

#####################################################################
############################   MODELING  PART 2 (after Feature Eng )    #######################
####################################################################

###############################     TRAIN-TEST SPLIT ##############


# Train test split
set.seed(42)  # Seed 


intrain <- createDataPartition(df$var_target, p = 0.8, list = FALSE)
#print(intrain)
training <- df[intrain,]
testing <- df[-intrain,]

###################### [TRAINING]   TARGET UNBALANCE ##############
table(training$var_target)  # distribution non bilanciata

# Undersampling
training_balanced <- training %>%
  group_by(var_target) %>%
  sample_n(min(table(training$var_target))) %>%
  ungroup()

table(training_balanced$var_target)  


colnames(training_balanced)


# Random forest dopo nuove features

rf_model_2 <- randomForest(var_target ~ .- id_cliente, data = training_balanced, ntree = 500)
#evaluate model and save to results dataframe
predictions_baseline_RF2 <- predict(rf_model_2, newdata = testing)
results_df <- rbind(eval_model('RF_baseline_2', testing,predictions_baseline_RF2),results_df)


predictions_rf2_perc <- predict(object = rf_model_2, 
                                  newdata = testing %>% select(-var_target), type='prob')[, "si"]

varImpPlot(rf_model_2,bg = hex_base)

# gain curve for boost model
gain_curve_rf2 <- gains(ifelse(testing$var_target == "si", 1, 0), 
                          as.numeric(factor(predictions_rf2_perc)), 
                          optimal = TRUE, percents = TRUE, groups=50)
gain_curve_rf2 <- bind_cols(c(0,gain_curve_rf2$depth), c(0,gain_curve_rf2$cume.pct.of.total*100)) %>% 
  rename(depth = ...1,
         cume.pct.of.total = ...2)


# boost tree

model_boost <- C5.0(x = training_balanced%>%select(-var_target, -id_cliente), 
                    y = as.factor(training_balanced$var_target), trials = 100)
predictions_boost <- predict(object = model_boost, 
                           newdata = testing %>% select(-id_cliente, -var_target)
                           )
# aggiungi a dataframe di risultati
results_df <- rbind(eval_model('model_boost', testing,predictions_boost),results_df)

predictions_boost_perc <- predict(object = model_boost, 
                                  newdata = testing %>% select(-id_cliente, -var_target), type='prob')[, "si"]
# gain curve for boost model
gain_curve_boost <- gains(ifelse(testing$var_target == "si", 1, 0), 
                    as.numeric(factor(predictions_boost_perc)), 
                    optimal = TRUE, percents = TRUE, groups=50)
gain_curve_boost <- bind_cols(c(0,gain_curve_boost$depth), c(0,gain_curve_boost$cume.pct.of.total*100)) %>% 
  rename(depth = ...1,
         cume.pct.of.total = ...2)


#plot(model_boost)

summary(model_boost)

#plot(model_boost, trial = 99 , subtree =11)



#####################################  CANDIDATE MODEL ############
# il modello boost sembra performare meglio

#####################################################################
#################   EVALUATION               #######################
####################################################################

#cumulative gain chart

#--------------- gain curves

#plot la gain curve

ggplot() +
  geom_line(data = gain_curve_boost, aes(x = depth, y = cume.pct.of.total, colour = "Boost"), show.legend = TRUE) +
  geom_line(data = gain_curve_rf2, aes(x = depth, y = cume.pct.of.total, colour = "RF2"), show.legend = TRUE) +
  geom_line(data = gain_curve_rf, aes(x = depth, y = cume.pct.of.total, colour = "RF"), show.legend = TRUE) +
  geom_line(data = gain_curve_tree, aes(x = depth, y = cume.pct.of.total, colour = "Tree"), show.legend = TRUE) +
  geom_line(data = gain_curve_naive, aes(x = depth, y = cume.pct.of.total, colour = "Naïve"), show.legend = TRUE) +
  geom_line(data = data.frame(x = c(0,100), y = c(0,100)), aes(x = x, y = y, colour = "Baseline"), show.legend = TRUE) +
  scale_color_manual(values = c("Boost" = hex_base, 
                                "RF2" = contrasting_palette[1], 
                                "RF" = contrasting_palette[2], 
                                "Tree" = contrasting_palette[3],
                                "Baseline" = "red", 
                                "Naïve"="black"))  +
  theme_minimal() +
  labs(title = "Cumulative Gains Chart", x = "% Customers Contacted", y = "% Positive Responses") +
  theme(legend.title = element_blank())

