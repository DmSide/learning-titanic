# -*- coding: utf-8 -*-
# Реализация в sklearn
# Различные метрики качества реализованы в пакете sklearn.metrics. Конкретные функции указаны в инструкции по выполнению задания.
#
# Материалы
# Подробнее о метриках качества
# Задачи по AUC-ROC


# *** Инструкция по выполнению ***

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
# 4. Имеется четыре обученных классификатора.
# В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
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
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
# В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
#
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.42. При необходимости округляйте дробную часть до двух знаков.
#
# Ответ на каждое задание — текстовый файл, содержащий ответ в первой строчке.
# Обратите внимание, что отправляемые файлы не должны содержать перевод строки в конце.
# Данный нюанс является ограничением платформы Coursera. Мы работаем над тем, чтобы убрать это ограничение.

import pandas
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, precision_recall_curve

if __name__ == '__main__':
    scores = pandas.read_csv('ecm_scores.csv')
    classification = pandas.read_csv('ecm_classification.csv')
    TP = len(classification[(classification.true == 1) & (classification.pred == 1)])
    FP = len(classification[(classification.true == 0) & (classification.pred == 1)])
    FN = len(classification[(classification.true == 1) & (classification.pred == 0)])
    TN = len(classification[(classification.true == 0) & (classification.pred == 0)])

    # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    # TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print(TP, FP, FN, TN)

    acc = accuracy_score(classification['true'], classification['pred'])
    prec = precision_score(classification['true'], classification['pred'])

    recall = recall_score(classification['true'], classification['pred'])
    f1 = f1_score(classification['true'], classification['pred'])

    print(acc, prec, recall, f1)

    score_logreg = roc_auc_score(scores['true'],scores['score_logreg'])
    score_svm = roc_auc_score(scores['true'], scores['score_svm'])
    score_knn = roc_auc_score(scores['true'], scores['score_knn'])
    score_tree = roc_auc_score(scores['true'], scores['score_tree'])

    print(score_logreg, score_svm, score_knn, score_tree)
    print(f'MAX: {max(score_logreg, score_svm, score_knn, score_tree)}')

    # sklearn.metrics.precision_recall_curve returns (precision, recall, thresholds)
    score_header_list = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
    max_prcs = []
    for score_header in score_header_list:
        print(score_header)
        prc = precision_recall_curve(scores['true'], scores[score_header])
        max_prc = max(prc[0][(prc[1] >= 0.7).nonzero()[0]])
        print(max_prc)
        max_prcs.append(max_prc)

    print(max(max_prcs))
    # prc_logreg = precision_recall_curve(scores['true'], scores['score_logreg'])
    # prc_svm = precision_recall_curve(scores['true'], scores['score_svm'])
    # prc_knn = precision_recall_curve(scores['true'], scores['score_knn'])
    # prc_tree = precision_recall_curve(scores['true'], scores['score_tree'])

    a = 56
    # (prc_logreg[1]  0.7).nonzero()[0]
