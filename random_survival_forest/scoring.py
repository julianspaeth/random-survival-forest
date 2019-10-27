from itertools import combinations


def concordance_index(y_time, y_pred, y_event):
    """
    Compute concordance index.
    :param y_time: Actual Survival Times.
    :param y_pred: Predicted cumulative hazard functions.
    :param y_event: Actual Survival Events.
    :return: c-index.
    """
    oob_predicted_outcome = [x.sum() for x in y_pred]
    possible_pairs = list(combinations(range(len(y_pred)), 2))
    concordance = 0
    permissible = 0
    for pair in possible_pairs:
        t1 = y_time.iat[pair[0]]
        t2 = y_time.iat[pair[1]]
        e1 = y_event.iat[pair[0]]
        e2 = y_event.iat[pair[1]]
        predicted_outcome_1 = oob_predicted_outcome[pair[0]]
        predicted_outcome_2 = oob_predicted_outcome[pair[1]]

        shorter_survival_time_censored = (t1 < t2 and e1 == 0) or (t2 < t1 and e2 == 0)
        t1_equals_t2_and_no_death = (t1 == t2 and (e1 == 0 and e2 == 0))

        if shorter_survival_time_censored or t1_equals_t2_and_no_death:
            continue
        else:
            permissible = permissible + 1
            if t1 != t2:
                if t1 < t2:
                    if predicted_outcome_1 > predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    elif predicted_outcome_1 == predicted_outcome_2:
                        concordance = concordance + 0.5
                        continue
                elif t2 < t1:
                    if predicted_outcome_2 > predicted_outcome_1:
                        concordance = concordance + 1
                        continue
                    elif predicted_outcome_2 == predicted_outcome_1:
                        concordance = concordance + 0.5
                        continue
            elif t1 == t2:
                if e1 == 1 and e2 == 1:
                    if predicted_outcome_1 == predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    else:
                        concordance = concordance + 0.5
                        continue
                elif not (e1 == 1 and e2 == 1):
                    if e1 == 1 and predicted_outcome_1 > predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    elif e2 == 1 and predicted_outcome_2 > predicted_outcome_1:
                        concordance = concordance + 1
                        continue
                    else:
                        concordance = concordance + 0.5
                        continue

    c = concordance / permissible

    return c
