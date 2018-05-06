import os

import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels


def get_callbacks(log_dir=None, valid=(), tensorboard=True, eary_stopping=True):
    """Get callbacks.

    Args:
        log_dir (str): the destination to save logs(for TensorBoard).
        valid (tuple): data for validation.
        tensorboard (bool): Whether to use tensorboard.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if log_dir and tensorboard:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)
        callbacks.append(TensorBoard(log_dir))

    if valid:
        callbacks.append(F1score(*valid))

    if log_dir:
        if not os.path.exists(log_dir):
            print('Successfully made a directory: {}'.format(log_dir))
            os.mkdir(log_dir)

        file_name = '_'.join(['model_weights', '{epoch:02d}', '{f1:2.2f}']) + '.h5'
        save_callback = ModelCheckpoint(os.path.join(log_dir, file_name),
                                        monitor='f1',
                                        save_weights_only=True)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=3, mode='max'))

    return callbacks


def get_entities(seq):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks


def f1_score(y_true, y_pred, sequence_lengths):
    """Evaluates f1 score.

    Args:
        y_true (list): true labels.
        y_pred (list): predicted labels.
        sequence_lengths (list): sequence lengths.

    Returns:
        float: f1 score.

    Example:
        >>> y_true = []
        >>> y_pred = []
        >>> sequence_lengths = []
        >>> print(f1_score(y_true, y_pred, sequence_lengths))
        0.8
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = set(get_entities(lab))
        lab_pred_chunks = set(get_entities(lab_pred))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return f1


class F1score(Callback):

    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor

    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        predictions = []
        truths = []

        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            sequence_lengths = data[-1] # shape of (batch_size, 1)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))
            #y_pred = np.asarray(self.model_.predict(data, sequence_lengths))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]

            for line in y_true:
                truths += line
            for line in y_pred:
                predictions += line
                
            a, b, c = self.count_correct_and_pred(y_true, y_pred, sequence_lengths)
            correct_preds += a
            total_preds += b
            total_correct += c
        # print ("\nTag-based evaluation.")
        # print (classification_report(truths, predictions, labels=unique_labels(predictions, truths)))
        print ("Chunk-based evaluation.")
        self.chunk_based_evaluate(truths, predictions)
        
        f1 = self._calc_f1(correct_preds, total_correct, total_preds)
        # print(' - f1: {:04.2f}'.format(f1 * 100))
        logs['f1'] = f1

    def chunk_based_evaluate(self, _actuals, _predictions):
        actuals = self.chunk(_actuals)
        predictions = self.chunk(_predictions)
        labels = list(set([e[0] for e in actuals]))
        print ("%10s%10s%11s%11s%10s" %
               (" ", "precision", "recall", "f1-score", "support"))
        avg_prec, avg_rec, avg_f1, total_support = 0, 0, 0, 0
        for lb in labels:
            truths = [e for e in actuals if e[0] == lb]
            preds = [e for e in predictions if e[0] == lb]
            corrects = [e for e in preds if e in truths]
            if len(preds) == 0:
                prec = 0
            else:
                prec = float(len(corrects)) / len(preds)
            if len(truths) == 0:
                rec = 0
            else:
                rec = float(len(corrects)) / len(truths)
            if prec + rec == 0:
                f1_score = 0
            else:
                f1_score = 2 * prec * rec / (prec + rec)
            avg_prec += prec * len(truths)
            avg_rec += rec * len(truths)
            avg_f1 += f1_score * len(truths)
            total_support += len(truths)
            print ("%10s%9.2f%11.2f%10.2f%10d" %
                   (lb, prec*100, rec*100, f1_score*100, len(truths)))
        print ("\n%10s%9.2f%10.2f%10.2f%10d" % ("avg / total", (avg_prec / total_support)*100,
                                                (avg_rec / total_support)*100, (avg_f1 / total_support)*100, total_support))

    def chunk(self, arr):
        entities = []
        for i in range(len(arr)):
            # end = None
            if arr[i][0].lower() == 'b':
                start = i
                if start < len(arr) - 1:
                    end = start + 1
                    while arr[end][0].lower() == 'i':
                        if end + 1 < len(arr):
                            end += 1
                        else:
                            end = end + 1
                            break
                    end = end - 1
                else:
                    end = start
                entities.append((arr[i][2:], start, end))
            # if end != None:
                # i = end
        return entities

    def _calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return f1

    def count_correct_and_pred(self, y_true, y_pred, sequence_lengths):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]

            lab_chunks = set(get_entities(lab))
            lab_pred_chunks = set(get_entities(lab_pred))
            # print ("lab_chunks", lab_chunks)
            # print ("lab_pred_chunks", lab_pred_chunks)

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
        return correct_preds, total_correct, total_preds
