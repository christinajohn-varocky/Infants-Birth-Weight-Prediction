# Birth_Weight_Prediction
# Introduction
A new born baby calls out for joy and hope, but the bitter truth is each year more than half a
million women have tragic pregnancies that result in anguish and despair. Underweight and
overweight babies have high mortality and morbidity rates. Detection of these growth
abnormalities will help us to avoid and manage perinatal complications. Risks in pregnancy
and miscarriage can be handled if the birth weight of the baby can be predicted before it is
born and hence, necessary changes in the mother’s lifestyle can be suggested, for the
betterment of the health of the baby as well as the mother. This study aims to predict baby
weight using different factors that affect the same.

# Model comparison

Using Machine Learning models to predict if the birth weight of a baby will fall outside the normal range (2.5kg-4.5kg). 

DATA DESCRIPTION:
The data was acquired from the National Centre for Health Statistics, a publicly available dataset, which contained 10,48,575 records and 55 attributes covering different aspects  and measures of pregnancy. The dataset contains details about the socio-economic factors of the parents, the mother's medical history and details about the pregnancy. The variable under study and to be predicted was Birth Weight (DBWT), ranging between 227-4700 (in grams). 

After implementing SMOTE to handle class imbalance, various classification models were built. The best performing model was CatBoost, with an AUC Score of 93.421%.
