#Programa de Pós-Graduação em Tecnologia, Gestão e Sustentabilidade - Instituto Federal de Goiás
#Disciplina "Sistema, Modelo e Simulação"
#Doscente: Prof. Édipo Henrique Cremon
#Discente: Gustavo Ferreira de Morais

##Site do banco de dados: https://www.kaggle.com/datasets/robjan/ph-recognition/code?resource=download


##Bibliotecas usadas:
library(corrplot)
library(caret)
library(gbm)
library(xgboost)
library(C50)
library(ggplot2)
library(Metrics)

##Selecao da pasta de trabalho
setwd("C:/Users/gusta/OneDrive/Documentos/R/MstSistModReg/TrabalhoFinal/Envio")


##Importando dados em .csv
ph <- read.csv("ph-data.csv", header = TRUE, sep = ",",  dec = ".")


##Chamando o visualizador de dados como tabela:
View(ph)

##Verificacao da estrutura dos dados:
str(ph)


#Matriz de correlacao dos dados
M <- cor(ph)
M

corrplot(M, method = "ellipse")
corrplot(M, method = "ellipse", type = "upper")
corrplot(M, method = "number", type = "upper")
corrplot.mixed(M, lower = "number", lower.col = "black", number.cex = .7, upper = "ellipse", tl.pos = "lt")
corrplot.mixed(M, lower = "number", number.cex = .7, upper = "ellipse", tl.pos = "lt")

##Importando dados em .csv classificados(acido de 1 a 6, neutro 7, basico de 8 a 14)
ph <- read.csv("ph-data_Tratado.csv", header = TRUE, sep = ",",  dec = ".")


##Converter a classe obito para categorica:
ph$pH <- as.character(ph$pH)


##Divisao das amostras em treinamento e validacao - 
##70% de amostras para treinamento e 30% de testes:
set.seed(100)
xvars <- as.vector(createDataPartition(ph$pH , list=FALSE, p=0.7))
treinamento <- ph[xvars,]
teste <- ph[-xvars,]

#Verificacao da quantidade de amostras de treinamento
nrow(treinamento)
nrow(teste)


##Elaborando modelos pelo pacote Caret:

#Modelo CART (rpart):
set.seed(100)
modelo_cart <- train(pH ~ .,  data = treinamento, method = "rpart",
                     trControl=trainControl(method = "CV", number = 10),
                     tuneLength = 10,
                     metric="Accuracy")
print(modelo_cart)

#Modelo Random forest (RF):
set.seed(100)
modelo_rf <- train(pH ~ .,  data = treinamento, method = "rf",
                   trControl=trainControl(method = "CV", number = 10),
                   metric="Accuracy")
print(modelo_rf)

#Modelo Gradient Boosting Machine (GBM):
set.seed(100)
modelo_gbm <- train(pH ~ .,  data = treinamento, method = "gbm",
                    trControl=trainControl(method = "CV", number = 10),
                    metric="Accuracy")
print(modelo_gbm)


#Modelo XGBoost
set.seed(100)
modelo_xgboost <- train(pH ~ .,  data = treinamento, method = "xgbTree",
                        trControl=trainControl(method = "CV", number = 10),
                        metric="Accuracy")
print(modelo_xgboost)


#Modelo C5.0
set.seed(100)
modelo_c50 <- train(pH ~ .,  data = treinamento, method = "C5.0",
                    trControl=trainControl(method = "CV", number = 10),
                    metric="Accuracy")
print(modelo_c50)


#Modelo Support Vector MAchine (SVM):
set.seed(100)
modelo_svm <- train(pH ~ .,  data = treinamento, method = "svmLinear",
                    trControl=trainControl(method = "CV", number = 10),
                    tuneLength = 10,
                    preProc = c("center", "scale"),
                    metric="Accuracy")
print(modelo_svm)


##Comparação entre os modelos gerados: 
comparacao <- resamples(list(CART=modelo_cart, RF=modelo_rf, GBM=modelo_gbm, XGBoost=modelo_xgboost, C5.0=modelo_c50, SVM=modelo_svm))
bwplot(comparacao)
summary(comparacao)


##Validação dos modelos gerados com dados de teste
#Matriz confusão e analises estatisticas nos dados dos modelos calculados:

#Modelo Cart:
CART_caret <- as.factor(predict(modelo_cart, teste, type="raw"))

teste$est_cart_caret <- CART_caret
CART_tabela2 <- table(teste$pH, CART_caret)
confusionMatrix(CART_tabela2)

#Modelo Random Forest
rf_caret <- as.factor(predict(modelo_rf, teste, type="raw"))

teste$est_rf_caret <- rf_caret
rf_tabela <- table(teste$pH, rf_caret)
confusionMatrix(rf_tabela)

#Modelo GBM
gbm_caret <- as.factor(predict(modelo_gbm, teste, type="raw"))

teste$est_gbm_caret <- gbm_caret
gbm_tabela <- table(teste$pH, gbm_caret)
confusionMatrix(gbm_tabela)

#Modelo XGBoost
xgboost_caret <- as.factor(predict(modelo_xgboost, teste, type="raw"))

teste$est_xgboost_caret <- xgboost_caret
xgboost_tabela <- table(teste$pH, xgboost_caret)
confusionMatrix(xgboost_tabela)

#Modelo C5.0
xgboost_c50 <- as.factor(predict(modelo_c50, teste, type="raw"))

teste$est_c50_caret <- xgboost_c50
c50_tabela <- table(teste$pH, xgboost_c50)
confusionMatrix(c50_tabela)

#Modelo SVM
SVM_caret <- as.factor(predict(modelo_svm, teste, type="raw"))

teste$est_svm_caret <- SVM_caret
SVM_tabela2 <- table(teste$pH, SVM_caret)
confusionMatrix(SVM_tabela2)

#Comparando as metricas
confusionMatrix(CART_tabela2)
confusionMatrix(rf_tabela)
confusionMatrix(gbm_tabela)
confusionMatrix(xgboost_tabela)
confusionMatrix(c50_tabela)
confusionMatrix(SVM_tabela2)

##Graficos de comparcao dos valores das metricas de treinamento e teste

##Importando dados em .csv
compAccuracy <- read.csv("CompAcurracy.csv", header = TRUE, sep = ",",  dec = ".")
compKappa <- read.csv("CompKappa.csv", header = TRUE, sep = ",",  dec = ".")
Dadosteste <-  read.csv("TrabalhoFinal.csv", header = TRUE, sep = ",",  dec = ".")

#Graficos

g1_Accuracy <- ggplot(compAccuracy) +
                   aes(x = Trein_Accur, y = Teste_Accur, colour = Modelos) +
                   geom_jitter(size = 3.25) +
                   scale_color_hue(direction = 1) +
                   labs(x = "Valores de Treinamento", y = "Valores de Teste", title = "Comparação dos Modelos", 
                   subtitle = "Acurácia") +
                   theme_gray() +
                   theme(plot.caption = element_text(size = 12L, face = "bold.italic"))
plot(g1_Accuracy)

g2_Kappa <- ggplot(compKappa) +
                aes(x = X0.8824685, y = X0.9723, colour = Cart) +
                geom_point(shape = "circle", size = 3.25) +
                scale_color_hue(direction = 1) +
                labs(x = "Valores de Treinamento", y = "Valores de Teste", title = "Comparação dos Modelos", 
                     subtitle = "Kappa", caption = "12", color = "Modelos") +
                theme_gray() +
                theme(plot.caption = element_text(size = 12L))

plot(g2_Kappa)


#Salvar o dado em csv
write.csv(teste, "C:/Users/gusta/OneDrive/Documentos/R/MstSistModReg/TrabalhoFinal/Envio/TrabalhoFinal.csv")
