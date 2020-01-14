# -*- coding: utf-8 -*-
# Реализация в sklearn
# Различные метрики качества реализованы в пакете sklearn.metrics. Конкретные функции указаны в инструкции по выполнению задания.
#
# Материалы
# Подробнее о метриках качества
# Задачи по AUC-ROC
# Инструкция по выполнению
# Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred).
# Заполните таблицу ошибок классификации:
# Actual Positive	Actual Negative
# Predicted Positive	TP	FP
# Predicted Negative	FN	TN
# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.
#
# 3. Посчитайте основные метрики качества классификатора:
#
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score
# В качестве ответа укажите эти четыре числа через пробел.
#
# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
#
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.
#
# 5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
#
# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
#
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds. В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
#
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.42. При необходимости округляйте дробную часть до двух знаков.
#
# Ответ на каждое задание — текстовый файл, содержащий ответ в первой строчке. Обратите внимание, что отправляемые файлы не должны содержать перевод строки в конце. Данный нюанс является ограничением платформы Coursera. Мы работаем над тем, чтобы убрать это ограничение.


if __name__ == '__main__':
    

