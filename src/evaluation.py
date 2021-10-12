import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
from sklearn import metrics
from sklearn import preprocessing


def evaluate(config, next_best_view, classification_sum_absolute_difference, classification_negative_entropy, classification_kl_divergence, tile_number):
    """
    Compute the next best view and evaluate its results in different conditions.
    :param config: Configurations from the config file (dictionary)
    :param next_best_view: The function to be used for computing the next best view method (function object)
    :param classification_sum_absolute_difference: The function to be used for computing sum of absolute difference of the frontal and tile classifications (function object)
    :param classification_negative_entropy: The function to be used for computing negative entropy of the tile classifications (function object)
    :param classification_kl_divergence: The function to be used for computing KL divergence of the frontal and tile classifications (function object)
    :param tile_number: Number of tiles in the tests (int)
    """

    # Set output address
    output_address = './results/'

    # Delete the old results
    for file in os.listdir(output_address):
        if (file[-3:] == 'jpg') or (file[-3:] == 'txt'):
            os.remove(os.path.join(output_address, file))

    # Set the names of the methods and metric names
    measure_names = ['Random'] + ['Histogram Variance', 'Surface Normal Score', 'Classification KL Divergence', 'Classification KL Divergence (single classifier)', 'Histogram Negative Entropy', 'Classification Negative Entropy', 'Classification SAD'] + ['Proposed NBV']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 score']

    # Get the list of object labels
    labels_data_frame = pd.read_csv('./model/labels.csv')
    labels = labels_data_frame.to_numpy()[:, labels_data_frame.columns.to_list().index('Labels')]

    # Set the classifier name
    if config['classifier_type'] == 'RF':
        classifier_name = 'Random Forest'
    elif config['classifier_type'] == 'SVM':
        classifier_name = 'Support Vector Machine'
    elif (config['classifier_type'] == 'NN') and (config['nn_network_architecture'] != 'resnet101'):
        classifier_name = 'Neural Network ' + str(config['nn_network_architecture'])
    elif (config['classifier_type'] == 'NN') and (config['nn_network_architecture'] == 'resnet101'):
        classifier_name = 'ResNet 101'
    else:
        raise Exception("Unknown classifier type " + str(config['classifier_type']) + ".")

    # Set the fusion function name
    fusion_type = config['decision_fusion_type']

    # Set the recognition threshold and confidence ratio and their sweep range
    assert config['confidence_ratio_sweep_max'] > 0, 'Confidence ratio sweep max is ' + str(config['confidence_ratio_sweep_max']) + ', but it must be greater than 0.'
    assert config['confidence_ratio_sweep_no'] >= 1, 'Confidence ratio sweep number is ' + str(config['confidence_ratio_sweep_no']) + ', but it must be at least 1.'
    assert config['recognition_threshold_sweep_no'] >= 1, 'Recognition threshold sweep number is ' + str(config['recognition_threshold_sweep_no']) + ', but it must be at least 1.'
    confidence_ratio_range = np.linspace(0, config['confidence_ratio_sweep_max'], config['confidence_ratio_sweep_no'])
    recognition_threshold_range = np.linspace(0, 1, config['recognition_threshold_sweep_no'])
    selected_confidence_ratio = config['confidence_ratio']
    selected_recognition_threshold = config['recognition_threshold']
    if not (selected_confidence_ratio in confidence_ratio_range):
        print('Warning: The selected confidence ratio ' + str(selected_confidence_ratio) + ' is not in the confidence ratio sweep values.\nAdding it to the sweep values...')
        confidence_ratio_range = np.sort(np.append(confidence_ratio_range, selected_confidence_ratio))
    if not (selected_recognition_threshold in recognition_threshold_range):
        print('Warning: The selected recognition threshold ' + str(selected_recognition_threshold) + ' is not in the recognition ratio sweep values.\nAdding it to the sweep values...')
        recognition_threshold_range = np.sort(np.append(recognition_threshold_range, selected_recognition_threshold))

    # Pre-define some arrays
    accuracy_matrix = np.zeros((confidence_ratio_range.size, recognition_threshold_range.size))
    precision_matrix = np.zeros((confidence_ratio_range.size, recognition_threshold_range.size))
    recall_matrix = np.zeros((confidence_ratio_range.size, recognition_threshold_range.size))
    f1_score_matrix = np.zeros((confidence_ratio_range.size, recognition_threshold_range.size))

    tpr_combined = dict()
    tpr_random = dict()
    tpr_frontal = dict()
    fpr_combined = dict()
    fpr_random = dict()
    fpr_frontal = dict()
    roc_auc_combined = dict()
    roc_auc_random = dict()
    roc_auc_frontal = dict()

    accuracy_per_confidence_threshold = np.array([])
    precision_per_confidence_threshold = np.array([])
    recall_per_confidence_threshold = np.array([])
    f1_score_per_confidence_threshold = np.array([])

    uncertain_count_per_threshold = np.array([])

    # Read the scores data from drive
    scores_data = np.load('./results/scores.npy', allow_pickle=True)

    # Get the total number of tests
    test_no = scores_data.shape[0]

    # Get the labels of columns of test data
    test_columns = pd.read_csv('./results/scores.csv').columns.to_list()

    # Sweep the confidence thresholds
    for recognition_i, recognition_threshold in enumerate(recognition_threshold_range):

        # Print the current recognition threshold
        print("Recognition threshold: " + str(recognition_threshold))

        for confidence_i, confidence_ratio in enumerate(confidence_ratio_range):

            # Print the current confidence ratio
            print("\tConfidence ratio: " + str(confidence_ratio))

            # Define some arrays afresh
            cum_prob_improvement = np.zeros(tile_number)
            all_fused_prob_ranked = np.zeros((0, len(measure_names), tile_number))
            all_fused_prob = np.zeros((0, len(measure_names), tile_number))
            # cum_fused_prob_ranked = np.zeros((len(measure_names), tile_number))
            # cum_fused_prob = np.zeros((len(measure_names), tile_number))
            uncertain_count = 0
            confusion_matrix = np.zeros((tile_number, len(measure_names), len(labels), len(labels)))  # side, measure, true, detected
            original_confusion_matrix = np.zeros((len(labels), len(labels)))

            all_frontal_predictions = np.zeros((test_no, len(labels)))
            all_combined_predictions = np.zeros((test_no, len(labels)))
            all_randomTile_predictions = np.zeros((test_no, len(labels)))
            all_ground_truths = np.zeros(test_no)

            for test in range(test_no):

                # Fetch the test results for the current test
                row = scores_data[test]

                # Check the test number of the current label is the same as the one in the csv file
                assert int(row[0]) == test, "The test numbers do not match!"

                # Read the label
                true_index = row[test_columns.index('Label')]

                # Extract the results
                front_probability = row[test_columns.index('Front Probability')]
                fused_probabilities = row[test_columns.index('Fused Probabilities')]
                tile_probabilities_dedicated = row[test_columns.index('Tile Probabilities (Dedicated Classifier)')]
                tile_probabilities_shared = row[test_columns.index('Tile Probabilities (Common Classifier)')]
                tile_uniformity = row[test_columns.index('Histogram Uniformity')]
                tile_variance = row[test_columns.index('Histogram Variance')]
                tile_negative_entropy = row[test_columns.index('Histogram Negative Entropy')]
                tile_surface_normal_score = row[test_columns.index('Surface Normal Score')]

                # Determine confidence on the frontal detection
                sorted_front_prob = sorted(front_probability, reverse=True)
                max_front_prob = sorted_front_prob[0]
                second_max_front_prob = sorted_front_prob[1]
                if second_max_front_prob != 0:
                    if (max_front_prob / second_max_front_prob) > confidence_ratio:
                        confident_front_detection = True
                    else:
                        confident_front_detection = False
                else:
                    confident_front_detection = True

                # Extract probabilities for the ground truth class
                front_gt_prob = front_probability[true_index]
                fused_gt_prob = [f[true_index] for f in fused_probabilities]

                # Compute the classification sum of absolute difference
                tile_classification_sad = np.array([classification_sum_absolute_difference(u, front_probability) for u in tile_probabilities_dedicated])
                tile_classification_sad = np.reshape(preprocessing.minmax_scale(tile_classification_sad), (tile_number,))

                # Compute the classification negative entropy
                tile_classification_negative_entropy = np.array([classification_negative_entropy(u) for u in tile_probabilities_dedicated])
                tile_classification_negative_entropy = np.reshape(preprocessing.minmax_scale(tile_classification_negative_entropy), (tile_number,))

                # Compute the classification KL divergence
                tile_classification_kl_divergence = np.array([classification_kl_divergence(u, front_probability) for u in tile_probabilities_dedicated])
                tile_classification_kl_divergence = np.reshape(preprocessing.minmax_scale(tile_classification_kl_divergence), (tile_number,))
                tile_classification_kl_divergence_shared = np.reshape(preprocessing.minmax_scale(np.array([classification_kl_divergence(u, front_probability) for u in tile_probabilities_shared])), (tile_number,))

                # Set the undecidedness of the tile classifications
                tile_undecidedness_dedicated = tile_classification_kl_divergence
                tile_undecidedness_shared = tile_classification_kl_divergence_shared

                # Rate tiles with the next best view method
                _, combined_measure = next_best_view((tile_variance, tile_surface_normal_score, tile_undecidedness_dedicated))

                # Random scoring of tiles
                tile_random = np.random.rand(tile_number)

                # Make the list of measure
                measures = [tile_random, tile_variance, tile_surface_normal_score, tile_undecidedness_dedicated, tile_undecidedness_shared, tile_negative_entropy, tile_classification_negative_entropy, tile_classification_sad, combined_measure]
                assert len(measure_names) == len(measures), "The metrics in the list \"measures\" must be equal to the ones in the list \"measure_names\"."

                # Save the ground truth label and the predicted label by the NBV measure, random active vision, and the frontal only classifier
                all_frontal_predictions[test] = front_probability
                all_combined_predictions[test] = fused_probabilities[np.argsort(combined_measure)[-1]]
                all_randomTile_predictions[test] = fused_probabilities[np.random.choice(np.arange(tile_number))]
                all_ground_truths[test] = true_index

                # Get the average ranking of probability improvement with the sorted use of the tiles based on the given metric
                if not confident_front_detection:
                    gt_improvement = np.maximum(np.array(fused_gt_prob) - front_gt_prob, -np.inf)
                    cum_prob_improvement += gt_improvement

                    fused_gt_improvement_idx = np.argsort(gt_improvement)

                    all_fused_prob = np.append(all_fused_prob, np.zeros((1, all_fused_prob.shape[1], all_fused_prob.shape[2])), axis=0)
                    all_fused_prob_ranked = np.append(all_fused_prob_ranked, np.zeros((1, all_fused_prob_ranked.shape[1], all_fused_prob_ranked.shape[2])), axis=0)
                    for measure_i, measure_mode in enumerate(measures):
                        sorted_measure_idx = np.argsort(measure_mode)

                        fused_prob_rank_ranked = [(gt_improvement.shape[0] - np.where(fused_gt_improvement_idx == smi)[0][0]) for smi in sorted_measure_idx]

                        all_fused_prob_ranked[-1, measure_i] = np.array(fused_prob_rank_ranked)

                        fused_prob_absolute_ranked = [gt_improvement[smi] for smi in sorted_measure_idx]
                        all_fused_prob[-1, measure_i] = np.array(fused_prob_absolute_ranked)

                    uncertain_count += 1.0

                # Compute the confusion matrices
                if np.max(front_probability) > recognition_threshold:
                    recognized_index = np.argmax(front_probability)
                else:
                    recognized_index = 0  # Backgorund class

                if confident_front_detection:
                    confusion_matrix[:, :, true_index, recognized_index] += 1
                else:
                    for measure_i, measure_mode in enumerate(measures):
                        for side_i, side in enumerate(np.argsort(measure_mode)):
                            # side = np.argmax(measure_mode)
                            current_fused_prob = fused_probabilities[side]

                            if np.max(current_fused_prob) > recognition_threshold:
                                recognized_index_fused = np.argmax(current_fused_prob)
                            else:
                                recognized_index_fused = 0  # Backgorund class

                            confusion_matrix[side_i, measure_i, true_index, recognized_index_fused] += 1
                original_confusion_matrix[true_index, recognized_index] += 1

            # Compute the performance metrics
            macro_average_precision = np.zeros((tile_number, len(measure_names), len(labels)))
            macro_average_recall = np.zeros((tile_number, len(measure_names), len(labels)))
            original_macro_average_precision = np.zeros((len(labels)))
            original_macro_average_recall = np.zeros((len(labels)))
            original_macro_average_false_positive_rate = np.zeros((len(labels)))
            accuracy_allTiles = np.zeros((tile_number, len(measure_names)))
            macro_average_false_positive_rate = np.zeros((tile_number, len(measure_names), len(labels)))

            for side_i in range(tile_number):
                for measure_i in range(len(measures)):
                    for conf_class in range(1, confusion_matrix.shape[2]):
                        num = confusion_matrix[side_i, measure_i, conf_class, conf_class]
                        denom = np.sum(confusion_matrix[side_i, measure_i, :, conf_class])
                        macro_average_precision[side_i, measure_i, conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

                        num = confusion_matrix[side_i, measure_i, conf_class, conf_class]
                        denom = np.sum(confusion_matrix[side_i, measure_i, conf_class, :])
                        macro_average_recall[side_i, measure_i, conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

                        untrue_classes = list(range(1, confusion_matrix.shape[2]))
                        untrue_classes.remove(conf_class)

                        num = np.sum(confusion_matrix[side_i, measure_i, untrue_classes, conf_class])
                        denom = np.sum(confusion_matrix[side_i, measure_i, untrue_classes, :])
                        macro_average_false_positive_rate[side_i, measure_i, conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

                    num = np.sum(confusion_matrix[(side_i, measure_i,) + np.diag_indices(confusion_matrix.shape[2])])
                    denom = np.sum(confusion_matrix[side_i, measure_i])
                    accuracy_allTiles[side_i, measure_i] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

            for conf_class in range(1, original_confusion_matrix.shape[1]):
                num = original_confusion_matrix[conf_class, conf_class]
                denom = np.sum(original_confusion_matrix[:, conf_class])
                original_macro_average_precision[conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

                num = original_confusion_matrix[conf_class, conf_class]
                denom = np.sum(original_confusion_matrix[conf_class, :])
                original_macro_average_recall[conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

                untrue_classes = list(range(1, original_confusion_matrix.shape[1]))
                untrue_classes.remove(conf_class)

                num = np.sum(original_confusion_matrix[untrue_classes, conf_class])
                denom = np.sum(original_confusion_matrix[untrue_classes, :])
                original_macro_average_false_positive_rate[conf_class] = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

            num = np.sum(original_confusion_matrix[np.diag_indices(original_confusion_matrix.shape[1])])
            denom = np.sum(original_confusion_matrix)
            original_accuracy = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))

            precision_allTiles = np.swapaxes(np.mean(macro_average_precision, axis=2), 0, 1)
            recall_allTiles = np.swapaxes(np.mean(macro_average_recall, axis=2), 0, 1)
            num = 2 * (precision_allTiles * recall_allTiles)
            denom = precision_allTiles + recall_allTiles
            f1_score_allTiles = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
            # false_positive_rate_allTiles = np.swapaxes(np.mean(macro_average_false_positive_rate, axis=2), 0, 1)
            accuracy_allTiles = np.swapaxes(accuracy_allTiles, 0, 1)

            precision = precision_allTiles[:, -1]
            recall = recall_allTiles[:, -1]
            f1_score = f1_score_allTiles[:, -1]
            # false_positive_rate = false_positive_rate_allTiles[:, -1]
            accuracy = accuracy_allTiles[:, -1]

            original_precision = np.mean(original_macro_average_precision)
            original_recall = np.mean(original_macro_average_recall)
            num = 2 * (original_precision * original_recall)
            denom = original_precision + original_recall
            original_f1_score = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
            # original_false_positive_rate = np.mean(original_macro_average_false_positive_rate)

            computed_metrics = [accuracy, precision, recall, f1_score]
            original_metrics = [original_accuracy, original_precision, original_recall, original_f1_score]
            computed_metrics_allTiles = [accuracy_allTiles, precision_allTiles, recall_allTiles, f1_score_allTiles]

            cum_fused_prob_ranked = np.mean(all_fused_prob_ranked, axis=0)
            cum_fused_prob = np.mean(all_fused_prob, axis=0)
            std_fused_prob_ranked = np.std(all_fused_prob_ranked, axis=0)
            std_fused_prob = np.std(all_fused_prob, axis=0)

            accuracy_matrix[confidence_i, recognition_i] = accuracy[-1]
            precision_matrix[confidence_i, recognition_i] = precision[-1]
            recall_matrix[confidence_i, recognition_i] = recall[-1]
            f1_score_matrix[confidence_i, recognition_i] = f1_score[-1]

            # Computing the ROC curves and area under the curve for the selected confidence threshold
            if confidence_ratio == selected_confidence_ratio:
                bin_gt = preprocessing.label_binarize(all_ground_truths, classes=np.arange(0, len(labels)))

                fpr_combined['micro'], tpr_combined['micro'], _ = metrics.roc_curve(bin_gt.ravel(), all_combined_predictions.ravel())
                roc_auc_combined['micro'] = metrics.auc(fpr_combined['micro'], tpr_combined['micro'])

                fpr_random['micro'], tpr_random['micro'], _ = metrics.roc_curve(bin_gt.ravel(), all_randomTile_predictions.ravel())
                roc_auc_random['micro'] = metrics.auc(fpr_random['micro'], tpr_random['micro'])

                fpr_frontal['micro'], tpr_frontal['micro'], _ = metrics.roc_curve(bin_gt.ravel(), all_frontal_predictions.ravel())
                roc_auc_frontal['micro'] = metrics.auc(fpr_frontal['micro'], tpr_frontal['micro'])

                for i in range(len(labels)):
                    fpr_combined[i], tpr_combined[i], _ = metrics.roc_curve(bin_gt[:, i], all_combined_predictions[:, i])
                    roc_auc_combined[i] = metrics.auc(fpr_combined[i], tpr_combined[i])

                    fpr_random[i], tpr_random[i], _ = metrics.roc_curve(bin_gt[:, i], all_randomTile_predictions[:, i])
                    roc_auc_random[i] = metrics.auc(fpr_random[i], tpr_random[i])

                    fpr_frontal[i], tpr_frontal[i], _ = metrics.roc_curve(bin_gt[:, i], all_frontal_predictions[:, i])
                    roc_auc_frontal[i] = metrics.auc(fpr_frontal[i], tpr_frontal[i])

            if recognition_threshold == selected_recognition_threshold:
                accuracy_per_confidence_threshold = np.append(accuracy_per_confidence_threshold, accuracy[-1])
                precision_per_confidence_threshold = np.append(precision_per_confidence_threshold, precision[-1])
                recall_per_confidence_threshold = np.append(recall_per_confidence_threshold, recall[-1])
                f1_score_per_confidence_threshold = np.append(f1_score_per_confidence_threshold, f1_score[-1])

                single_original_accuracy = original_accuracy
                single_original_precision = original_precision
                single_original_recall = original_recall
                single_original_f1_score = original_f1_score

                single_original_metrics = original_metrics

            # In case of the selected confidence ratio
            if (confidence_ratio == selected_confidence_ratio) and (recognition_threshold == selected_recognition_threshold):

                if not os.path.exists(output_address):
                    os.mkdir(output_address)

                # Plot the information
                for std_data in zip([None, std_fused_prob_ranked], [None, std_fused_prob], ['', '_with_std']):
                    # Ranked improvement per ordered tile
                    plt.figure(figsize=(15, 15))
                    plt.suptitle('Average ranked improvement of ordered tiles\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                    for i, elem in enumerate(cum_fused_prob_ranked[1:]):
                        plt.subplot(3, 3, i + 1)
                        if std_data[0] is None:
                            plt.bar(range(1, tile_number + 1), elem, color='g')
                        else:
                            plt.bar(range(1, tile_number + 1), elem, color='g', yerr=std_data[0][i])
                        plt.ylim([1, 5])
                        plt.title(str(measure_names[i + 1]))
                        plt.xlabel('Tile order (ascending sort of measure magnitude)')
                        plt.ylabel('Average rank of probability improvement of the ground truth')

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.93)
                    plt.savefig(os.path.join(output_address, 'ranked_imp' + std_data[2] + '.jpg'), dpi=200)
                    plt.close()

                    # Absolute improvements per ordered tile
                    plt.figure(figsize=(15, 15))
                    plt.suptitle('Average absolute improvement of ordered tiles\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                    for i, elem in enumerate(cum_fused_prob[1:]):
                        plt.subplot(3, 3, i + 1)
                        if std_data[1] is None:
                            plt.bar(range(1, tile_number + 1), elem, color='g')
                        else:
                            plt.bar(range(1, tile_number + 1), elem, color='g', yerr=std_data[1][i])
                        plt.ylim([-0.1, 0.4])
                        plt.title(str(measure_names[i + 1]))
                        plt.xlabel('Tile order (ascending sort of measure magnitude)')
                        plt.ylabel('Average probability improvement of the ground truth')

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.93)
                    plt.savefig(os.path.join(output_address, 'absolute_imp' + std_data[2] + '.jpg'), dpi=200)
                    plt.close()

                # Confusion matrices
                plt.figure(figsize=(15, 15))
                plt.suptitle('Confusion matrices from different measures\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                for i, elem in enumerate(confusion_matrix[-1, 1:]):
                    plt.subplot(3, 3, i + 1)
                    df_cm = pd.DataFrame(elem.astype(np.int), range(elem.shape[0]), range(elem.shape[0]))
                    sn.set(font_scale=1.0)
                    sn.heatmap(df_cm, annot=True, cbar=False, cmap="Greens", fmt="d", square=True, annot_kws={"size": 10})
                    plt.title(str(measure_names[i + 1]))
                    plt.xticks(np.array(range(len(labels))) + 0.5, labels, rotation='vertical')
                    plt.yticks(np.array(range(len(labels))) + 0.5, labels, rotation='horizontal')

                plt.subplot(3, 3, confusion_matrix.shape[1])
                df_cm = pd.DataFrame(original_confusion_matrix.astype(np.int), range(original_confusion_matrix.shape[0]), range(original_confusion_matrix.shape[0]))
                sn.set(font_scale=1.0)
                sn.heatmap(df_cm, annot=True, cbar=False, cmap="Greens", fmt="d", square=True, annot_kws={"size": 10})
                plt.title('No active vision')
                plt.xticks(np.array(range(len(labels))) + 0.5, labels, rotation='vertical')
                plt.yticks(np.array(range(len(labels))) + 0.5, labels, rotation='horizontal')

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                plt.savefig(os.path.join(output_address, 'confusion_matrices.jpg'), dpi=200)
                plt.close()

                # Performance metrics improvement per measure
                computed_metrics_improvement = [accuracy - original_accuracy, precision - original_precision, recall - original_recall, f1_score - original_f1_score]
                plt.figure(figsize=(15, 15))
                plt.suptitle('Performance metrics improvement per measure\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')
                for i, elem in enumerate(computed_metrics_improvement):
                    plt.subplot(2, 2, i + 1)
                    bar_list = plt.bar(range(1, len(measure_names) + 1), 100 * elem, color='g')
                    bar_list[-1].set_color('r')
                    plt.ylim([-5, 35])
                    plt.title(str(metric_names[i]) + ' improvement in different tiling-based next best view techniques')
                    plt.xticks(range(1, len(measure_names) + 1), measure_names, rotation='vertical')
                    plt.ylabel(metric_names[i] + ' improvement (%)')

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                plt.savefig(os.path.join(output_address, 'metrics_improvement_per_measure.jpg'), dpi=200)
                plt.close()

                # Performance metrics improvement per ordered tile
                plt.figure(figsize=(15, 15))
                plt.suptitle('Performance metrics improvement of ordered tiles\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                for i in range(1, len(measures)):
                    plt.subplot(3, 3, i)
                    plt.bar(np.arange(1, tile_number + 1), 100 * (precision_allTiles[i] - original_precision), width=0.2, color='y', label='Precision')
                    plt.bar(np.arange(1, tile_number + 1) + 0.2, 100 * (f1_score_allTiles[i] - original_f1_score), width=0.2, color='c', label='F1 Score')
                    plt.bar(np.arange(1, tile_number + 1) + 0.4, 100 * (recall_allTiles[i] - original_recall), width=0.2, color='m', label='Recall')
                    plt.bar(np.arange(1, tile_number + 1) + 0.6, 100 * (accuracy_allTiles[i] - original_accuracy), width=0.2, color='g', label='Accuracy')

                    plt.ylim([-5, 35])
                    plt.title(measure_names[i])
                    plt.ylabel('Improvement (%)')
                    plt.xlabel('Tile order (ascending sort of measure magnitude)')
                    plt.legend()

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                plt.savefig(os.path.join(output_address, 'ranked_metrics_imp.jpg'), dpi=200)
                plt.close()

                # Performance metrics per measure
                plt.figure(figsize=(15, 15))
                plt.suptitle('Performance metrics per measure\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                for i, elem in enumerate(computed_metrics):
                    plt.subplot(2, 2, i + 1)
                    bar_list = plt.bar(range(1, len(measure_names) + 2), 100 * np.concatenate((np.array([original_metrics[i]]), elem)), color='g')
                    bar_list[-1].set_color('r')
                    bar_list[0].set_color('b')
                    plt.ylim([0, 100])
                    plt.title(str(metric_names[i]) + ' in different tiling-based next best view techniques')
                    plt.xticks(range(1, len(measure_names) + 2), ['No active vision'] + measure_names, rotation='vertical')
                    plt.yticks(range(0, 101, 5))
                    plt.ylabel(metric_names[i] + ' (%)')

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                plt.savefig(os.path.join(output_address, 'metrics_per_measure.jpg'), dpi=200)
                plt.close()

                # Performance metrics per measure
                plt.figure(figsize=(15, 15))
                plt.suptitle('Performance metrics per measure\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

                for i, elem in enumerate(computed_metrics):
                    plt.subplot(1, 4, i + 1)
                    bar_heights = 100 * np.append(np.array([original_metrics[i]]), (elem[0], elem[-1]))
                    bar_list = plt.bar(range(1, 1 + bar_heights.size), bar_heights, color='g')
                    bar_list[-1].set_color('r')
                    bar_list[0].set_color('b')
                    plt.ylim([0, 100])
                    plt.title(str(metric_names[i]), fontsize='large')
                    plt.xticks(range(1, 1 + bar_heights.size), ['No active vision'] + ['Random active vision'] + [measure_names[-1]], rotation='vertical', fontsize='large')
                    plt.yticks(range(0, 101, 5))
                    plt.ylabel(metric_names[i] + ' (%)', fontsize='large')
                    for i, height in enumerate(bar_heights):
                        plt.text(i + 0.9, height + 1, '{:.2f}'.format(height), fontsize='large')

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)
                plt.savefig(os.path.join(output_address, 'main_metrics_per_measure.jpg'), dpi=200)
                plt.close()

                # Saving Numpy arrays in the selected confidence ratio
                results_dict = {'ranked improvement per ordered tile': cum_fused_prob_ranked,
                                'absolute improvements per ordered tile': cum_fused_prob,
                                'confusion matrices for each measure': confusion_matrix[-1],
                                'original confusion matrix': original_confusion_matrix,
                                'Original performance metrics': dict(zip(metric_names, original_metrics)),
                                'performance metrics improvement per measure': dict(zip(metric_names, computed_metrics_improvement)),
                                'performance metrics per measure': dict(zip(metric_names, computed_metrics)),
                                'performance metrics of all sorted tiles per measure': dict(zip(metric_names, computed_metrics_allTiles))}

                # Printing information in the selected confidence ratio
                text_output = ['Classifier: ' + classifier_name + ', Fusion: ' + fusion_type]
                text_output.append('\n\n\nSelected confidence threshold: {}, recognition threshold: {}. Uncertain detections count: {}'.format(selected_confidence_ratio, selected_recognition_threshold, int(uncertain_count)))

                text_output.append("\n\n\nAverage improvement ranking of ordered tiles:\n")
                for i, single_measurement in enumerate(cum_fused_prob_ranked):
                    text_output.append('\n{}: {}'.format(measure_names[i], single_measurement))

                text_output.append('\n\n\nAverage absolute improvement of ordered tiles:\n')
                for i, single_measurement in enumerate(cum_fused_prob):
                    text_output.append('\n{}: {}'.format(measure_names[i], single_measurement))

                text_output.append("\n\n\nPerformance metrics improvement per measure:\n")
                for i, single_measurement in enumerate(computed_metrics_improvement):
                    text_output.append("\n{}: {}".format(metric_names[i], single_measurement))

                text_output.append("\n\n\nPerformance metrics per measure:\n")
                for i, single_measurement in enumerate(computed_metrics):
                    text_output.append("\n{}: {}, Original: {}".format(metric_names[i], single_measurement, original_metrics[i]))

                text_output_temp = ['\n\n\nConfusion matrices for each measure:\n']
                text_output_temp.append('\nOriginal:\n\n{}'.format(original_confusion_matrix))
                for i, single_measurement in enumerate(confusion_matrix[-1]):
                    text_output_temp.append('\n\n{}:\n{}'.format(measure_names[i], single_measurement))

            # Save the current uncertain count
            if recognition_threshold == selected_recognition_threshold:
                uncertain_count_per_threshold = np.append(uncertain_count_per_threshold, uncertain_count)

    # Plot performance metrics improvement per confidence threshold
    plt.figure(figsize=(15, 15))
    plt.suptitle('Performance metrics improvement per confidence threshold\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

    computed_metrics_improvement_per_confidence_threshold = [accuracy_per_confidence_threshold - single_original_accuracy, precision_per_confidence_threshold - single_original_precision, recall_per_confidence_threshold - single_original_recall,
                                                             f1_score_per_confidence_threshold - single_original_f1_score]

    for i, elem in enumerate(computed_metrics_improvement_per_confidence_threshold):
        axes = plt.subplot(2, 2, i + 1)
        plt.plot(confidence_ratio_range, 100 * elem, linewidth=5, color='g', marker='o')
        plt.ylim([-5, 35])
        plt.xlim([0, int(confidence_ratio_range[-1] * 1.05)])
        axes.set_facecolor((0.95, 0.95, 0.95))
        plt.title(str(metric_names[i]) + ' improvement per confidence threshold')
        plt.ylabel(metric_names[i] + ' improvement (%)')
        plt.xlabel('Confidence threshold')
        plt.yticks(np.arange(-5, 36, 5))

        for spine in axes.spines.values():
            spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(output_address, 'metrics_improvement_per_confidence_threshold.jpg'), dpi=200)
    plt.close()

    # Plot performance metrics per confidence threshold
    plt.figure(figsize=(15, 15))
    plt.suptitle('Performance metrics per confidence threshold\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

    computed_metrics_per_confidence_threshold = [accuracy_per_confidence_threshold, precision_per_confidence_threshold, recall_per_confidence_threshold, f1_score_per_confidence_threshold]

    for i, elem in enumerate(computed_metrics_per_confidence_threshold):
        axes = plt.subplot(2, 2, i + 1)
        plt.plot(confidence_ratio_range, 100 * elem, linewidth=5, color='g', marker='o')
        plt.ylim([0, 100])
        plt.xlim([0, int(confidence_ratio_range[-1] * 1.05)])
        axes.set_facecolor((0.95, 0.95, 0.95))
        plt.title(str(metric_names[i]) + ' per confidence threshold')
        plt.ylabel(metric_names[i] + ' (%)')
        plt.xlabel('Confidence threshold')
        plt.yticks(np.arange(0, 101, 5))

        for spine in axes.spines.values():
            spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(output_address, 'metrics_per_confidence_threshold.jpg'), dpi=200)
    plt.close()

    # Plot performance metrics improvement over fusion effort per confidence threshold
    plt.figure(figsize=(15, 15))
    plt.suptitle('Ratio of performance metrics improvement to fusion efforts per confidence threshold\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

    for i, elem in enumerate(computed_metrics_improvement_per_confidence_threshold):
        axes = plt.subplot(2, 2, i + 1)
        ratio = elem[uncertain_count_per_threshold > 0]
        ratio /= uncertain_count_per_threshold[uncertain_count_per_threshold > 0]
        x_array = np.array(confidence_ratio_range)[uncertain_count_per_threshold > 0]
        plt.plot(x_array, ratio / np.max(ratio), linewidth=5, color='g', marker='o')
        plt.ylim([0, 2])
        plt.xlim([x_array[0], int(x_array[-1] * 1.05)])
        axes.set_facecolor((0.95, 0.95, 0.95))
        plt.title('Normalized ' + str(metric_names[i]).lower() + ' improvement based on fusion effort per confidence threshold')
        plt.ylabel('Normalized ' + metric_names[i].lower() + ' improvement based on fusion attemps')
        plt.xlabel('Confidence threshold')
        plt.yticks(np.linspace(0, 2, 21))

        for spine in axes.spines.values():
            spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(output_address, 'metrics_impr_uncertains_ratio_per_confidence_threshold.jpg'), dpi=200)
    plt.close()

    # Plot performance metrics over fusion effort per confidence threshold
    plt.figure(figsize=(15, 15))
    plt.suptitle('Ratio of performance metrics to fusion efforts per confidence threshold\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

    for i, elem in enumerate(computed_metrics_per_confidence_threshold):
        axes = plt.subplot(2, 2, i + 1)
        ratio = elem[uncertain_count_per_threshold > 0]
        ratio /= uncertain_count_per_threshold[uncertain_count_per_threshold > 0]
        x_array = np.array(confidence_ratio_range)[uncertain_count_per_threshold > 0]
        plt.plot(x_array, ratio / np.max(ratio), linewidth=5, color='g', marker='o')
        plt.ylim([0, 2])
        plt.xlim([x_array[0], int(x_array[-1] * 1.05)])
        axes.set_facecolor((0.95, 0.95, 0.95))
        plt.title('Normalized ' + str(metric_names[i]).lower() + ' based on fusion effort per confidence threshold')
        plt.ylabel('Normalized ' + metric_names[i].lower() + ' based on fusion attemps')
        plt.xlabel('Confidence threshold')
        plt.yticks(np.linspace(0, 2, 21))

        for spine in axes.spines.values():
            spine.set_color('black')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(output_address, 'metrics_uncertains_ratio_per_confidence_threshold.jpg'), dpi=200)
    plt.close()

    # Plot ROC curve in the selected confidence threshold
    plt.figure(figsize=(15, 15))
    plt.suptitle('ROC curves\nClassifier: ' + classifier_name + ', Fusion: ' + fusion_type, fontsize='x-large')

    ax = plt.subplot(3, 4, 1, aspect=1)
    plt.plot(fpr_combined['micro'], tpr_combined['micro'], linewidth=3, markersize=1, color='r', marker='o', label='NBV fusion (AUC: ' + "{:.3f}".format(roc_auc_combined['micro']) + ')')
    plt.plot(fpr_frontal['micro'], tpr_frontal['micro'], linewidth=3, markersize=1, color='b', marker='o', label='No fusion (AUC: ' + "{:.3f}".format(roc_auc_frontal['micro']) + ')')
    plt.plot(fpr_random['micro'], tpr_random['micro'], linewidth=3, markersize=1, color='g', marker='o', label='Random fusion (AUC: ' + "{:.3f}".format(roc_auc_random['micro']) + ')')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.title('Micro-averaging ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('Recall')
    ax.set_facecolor((0.95, 0.95, 0.95))
    plt.legend(loc='lower right')

    for (i, current_label) in zip(np.arange(1, len(labels)), labels[1:]):
        ax = plt.subplot(3, 4, i + 1, aspect=1)
        plt.plot(fpr_combined[i], tpr_combined[i], linewidth=3, markersize=1, color='r', marker='o', label='NBV fusion (AUC: ' + "{:.3f}".format(roc_auc_combined[i]) + ')')
        plt.plot(fpr_frontal[i], tpr_frontal[i], linewidth=3, markersize=1, color='b', marker='o', label='No fusion (AUC: ' + "{:.3f}".format(roc_auc_frontal[i]) + ')')
        plt.plot(fpr_random[i], tpr_random[i], linewidth=3, markersize=1, color='g', marker='o', label='Random fusion (AUC: ' + "{:.3f}".format(roc_auc_random[i]) + ')')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.title('ROC curve for the class ' + current_label)
        plt.xlabel('False positive rate')
        plt.ylabel('Recall')
        ax.set_facecolor((0.95, 0.95, 0.95))
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(output_address, 'roc_curve.jpg'), dpi=200)
    plt.close()

    # Plot the 3D surface plot of performance metrics over different recognition and confidence thresholds
    if recognition_threshold_range.size > 1:
        computed_metrics_matrices = [accuracy_matrix, precision_matrix, recall_matrix, f1_score_matrix]

        x_3d, y_3d = np.meshgrid(recognition_threshold_range, confidence_ratio_range)

        for i, elem in enumerate(computed_metrics_matrices):
            ax3d = plt.subplot(2, 2, i + 1, projection='3d')
            ax3d.plot_surface(x_3d, y_3d, elem, rstride=1, cstride=1, cmap='viridis')
            ax3d.set_title(metric_names[i] + ' over different confidence and recognition thresholds', fontdict={'fontsize': 6})
            ax3d.set_xlabel('Recognition threshold', fontsize=5)
            ax3d.set_ylabel('Confidence threshold', fontsize=5)
            ax3d.set_zlabel(metric_names[i], fontsize=5)
            ax3d.set_xticks(recognition_threshold_range)
            ax3d.set_yticks(confidence_ratio_range)
            ax3d.set_zticks([0, 1, 0.1])
            ax3d.xaxis.set_tick_params(labelsize=3)
            ax3d.yaxis.set_tick_params(labelsize=3)
            ax3d.zaxis.set_tick_params(labelsize=3)

        plt.subplots_adjust(top=0.93)
        plt.savefig(os.path.join(output_address, 'metrics_thresholds_3d.jpg'), dpi=200)
        plt.close()

    # Save Numpy array for metrics per confidence threshold information
    results_dict['performance metrics improvement per confidence threshold'] = dict(zip(metric_names, computed_metrics_improvement_per_confidence_threshold))
    results_dict['performance metrics per confidence threshold'] = dict(zip(metric_names, computed_metrics_per_confidence_threshold))

    with open(os.path.join(output_address, 'Dicts_Arrays.pkl'), 'wb') as file:
        pickle.dump(results_dict, file)

    # Print metrics per confidence threshold information
    text_output.append('\n\n\nPerformance metrics improvement per confidence threshold:\n')
    for i, single_measurement in enumerate(computed_metrics_improvement_per_confidence_threshold):
        text_output.append("\n{}: {}".format(metric_names[i], single_measurement))

    text_output.append('\n\n\nPerformance metrics per confidence threshold:\n')
    for i, single_measurement in enumerate(computed_metrics_per_confidence_threshold):
        text_output.append("\n{}: {}, Original: {}".format(metric_names[i], single_measurement, single_original_metrics[i]))

    text_output.append('\n\n\nAverage Tile Improvement: {}'.format(cum_prob_improvement / test_no))

    text_output = text_output + text_output_temp

    with open(os.path.join(output_address, 'printed_results.txt'), 'w') as text_file:
        text_file.writelines(text_output)
    print(("{}\n" * len(text_output)).format(*text_output))
