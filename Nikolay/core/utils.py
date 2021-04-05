import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

def parse_odds(df):
    """
    Parse odds from avgOdds
    on fighters_df
    """

    # Добавить кэффы в df
    for i in df.index[:]:
        avgodds = df.loc[i, 'avgOdds']

        if avgodds == '[]':
            continue

        # Преобразую данные о коэффах из строки
        ids1, odd1 = int(avgodds.split()[1].split(',')[0]), float(avgodds.split()[3].split('}')[0])
        ids2, odd2 = int(avgodds.split()[5].split(',')[0]), float(avgodds.split()[7].split('}')[0])

        # Проверить совпадает ли fighterId_1 с ids1 (который в avgOdds)
        if df.loc[i, 'fighterId_1'] == ids1:
            df.at[i, 'odd1'] = odd1
            df.at[i, 'odd2'] = odd2

        if df.loc[i, 'fighterId_1'] == ids2:
            df.at[i, 'odd1'] = odd2
            df.at[i, 'odd2'] = odd1

    return df.drop('avgOdds', axis=1)


def combine_df(to_combine_df, fighter1_cols, fighter2_cols, fightStats_cols):
    df_combined = pd.DataFrame(to_combine_df[fighter1_cols].values - to_combine_df[fighter2_cols].values,
                               index=to_combine_df.index)
    df_combined.columns = fightStats_cols

    # df_combined = df_combined.fillna(0)
    df_combined['eventDate.date'] = to_combine_df['eventDate.date']
    df_combined['winner'] = to_combine_df['winner']
    df_combined['odd_diff'] = to_combine_df['odd1'] - to_combine_df['odd2']
    df_combined['fightsAmount_diff'] = to_combine_df['fighter1_fightsAmount'] - to_combine_df['fighter2_fightsAmount']
    df_combined['odd_diff'] = to_combine_df['odd1'] - to_combine_df['odd2']

    df_combined['fighter1_fightsAmount'] = to_combine_df['fighter1_fightsAmount']
    df_combined['fighter2_fightsAmount'] = to_combine_df['fighter2_fightsAmount']

    df_combined['odd1'] = to_combine_df['odd1']
    df_combined['odd2'] = to_combine_df['odd2']

    #     df_combined = df_combined[~df_combined['odd_diff'].isna()]
    #     to_combine_df = to_combine_df[(to_combine_df['fighter1_fightsAmount'] > 4) & (to_combine_df['fighter2_fightsAmount'] > 4)]
    return df_combined

def get_winner_favorite(df):
    favorite_list = []
    for odd1, odd2 in df[['f1_odds', 'f2_odds']].values:
        if odd1 < odd2: favorite = True
        else:           favorite = False

        favorite_list.append(favorite)

    df['favorite'] = favorite_list
    return df


def calculate_roi(df):
    bet = 100  # рублей
    bank = 0
    bet_res = []
    for ind in tqdm(df.index):
        winner, y_pred, odd1, odd2 = df.loc[ind, ['winner', 'y_pred', 'f1_odds', 'f2_odds']]

        if winner == y_pred:
            if winner:
                win_odd = odd1
            else:
                win_odd = odd2

            bank += bet * win_odd - bet
            bet_res.append(bet * win_odd - bet)

        else:
            bank -= bet
            bet_res.append(-bet)

    output = {
        'bets_count:': df.shape[0],
        'finish_bank:': bank,
        'roi:': np.mean(bet_res),
        'accuracy:': (df['winner'] == df['y_pred']).mean()
    }
    return output


def last_figher_id(fighterId, df):
    eventId_date_1 = df[df['fighterId_1'] == fighterId]['eventDate.date'].max()
    eventId_date_2 = df[df['fighterId_2'] == fighterId]['eventDate.date'].max()
    # If no fighters stats in the past rise error
    if type(eventId_date_1) == float and type(eventId_date_2) == float:
        rise_error = True
    else:
        rise_error = False
    if np.nan in [eventId_date_1, eventId_date_2]:
        if type(eventId_date_1) == float:
            use_id = 2

        if type(eventId_date_2) == float:
            use_id = 1
    else:
        use_id = np.argmax([eventId_date_1, eventId_date_2]) + 1

    return use_id, rise_error


def parse_data_from_fight(fightStats, duration):
    if len(fightStats) == 0:
        return [np.nan] * 23

    hitsTotal = fightStats.get('hitsTotal')
    hitsSuccessful = fightStats.get('hitsSuccessful')
    accentedHitsTotal = fightStats.get('accentedHitsTotal')
    accentedHitsSuccessful = fightStats.get('accentedHitsSuccessful')
    takedownTotal = fightStats.get('takedownTotal')
    takedownSuccessful = fightStats.get('takedownSuccessful')
    accentedHitsPositionDistanceTotal = fightStats.get('accentedHitsPositionDistanceTotal')
    accentedHitsPositionDistanceSuccessful = fightStats.get('accentedHitsPositionDistanceSuccessful')
    accentedHitsPositionClinchTotal = fightStats.get('accentedHitsPositionClinchTotal')
    accentedHitsPositionClinchSuccessful = fightStats.get('accentedHitsPositionClinchSuccessful')
    accentedHitsPositionParterTotal = fightStats.get('accentedHitsPositionParterTotal')
    accentedHitsPositionParterSuccessful = fightStats.get('accentedHitsPositionParterSuccessful')

    try:
        hitsSuccessful_percent = hitsSuccessful / hitsTotal
    except ZeroDivisionError:
        hitsSuccessful_percent = np.nan

    try:
        accentedHitsSuccessful_percent = accentedHitsSuccessful / hitsTotal
    except ZeroDivisionError:
        accentedHitsSuccessful_percent = np.nan

    try:
        accentedHits_percent = accentedHitsTotal / hitsTotal
    except ZeroDivisionError:
        accentedHits_percent = np.nan

    try:
        takedownSuccessful_percent = takedownSuccessful / takedownTotal
    except ZeroDivisionError:
        takedownSuccessful_percent = np.nan

    try:
        accentedHitsPositionDistanceSuccessful_percent = accentedHitsPositionDistanceTotal / accentedHitsPositionDistanceSuccessful
    except ZeroDivisionError:
        accentedHitsPositionDistanceSuccessful_percent = np.nan

    try:
        accentedHitsPositionClinchSuccessful_percent = accentedHitsPositionClinchTotal / accentedHitsPositionClinchSuccessful
    except ZeroDivisionError:
        accentedHitsPositionClinchSuccessful_percent = np.nan

    try:
        accentedHitsPositionParterSuccessful_percent = accentedHitsPositionParterTotal / accentedHitsPositionParterSuccessful
    except ZeroDivisionError:
        accentedHitsPositionParterSuccessful_percent = np.nan

    try:
        takedowns_to_hits = takedownSuccessful / hitsSuccessful
    except ZeroDivisionError:
        takedowns_to_hits = np.nan

    try:
        HitsPositionDistance_to_hits = accentedHitsPositionDistanceSuccessful / hitsSuccessful
    except ZeroDivisionError:
        HitsPositionDistance_to_hits = np.nan

    try:
        HitsPositionClinch_to_hits = accentedHitsPositionClinchSuccessful / hitsSuccessful
    except ZeroDivisionError:
        HitsPositionClinch_to_hits = np.nan

    try:
        HitsPositionParter_to_hits = accentedHitsPositionParterSuccessful / hitsSuccessful
    except ZeroDivisionError:
        HitsPositionParter_to_hits = np.nan

    hitsPM = (60 * hitsTotal) / duration
    accentedHitsPM = (60 * accentedHitsTotal) / duration
    takedownsPM = (60 * takedownTotal) / duration
    accentedHitsDistancePM = (60 * accentedHitsPositionDistanceTotal) / duration
    accentedHitsClinchPM = (60 * accentedHitsPositionClinchTotal) / duration
    accentedHitsParterPM = (60 * accentedHitsPositionParterTotal) / duration

    hitsSuccessfulPM = (60 * hitsSuccessful) / duration
    accentedHitsSuccessfulPM = (60 * accentedHitsSuccessful) / duration
    takedownsSuccessfulPM = (60 * takedownSuccessful) / duration
    accentedHitsDistanceSuccessfulPM = (60 * accentedHitsPositionDistanceSuccessful) / duration
    accentedHitsClinchSuccessfulPM = (60 * accentedHitsPositionClinchSuccessful) / duration
    accentedHitsParterSuccessfulPM = (60 * accentedHitsPositionParterSuccessful) / duration

    return hitsPM, accentedHitsPM, takedownsPM, \
           accentedHitsDistancePM, accentedHitsClinchPM, accentedHitsParterPM, \
           hitsSuccessfulPM, accentedHitsSuccessfulPM, takedownsSuccessfulPM, \
           accentedHitsDistanceSuccessfulPM, accentedHitsClinchSuccessfulPM, accentedHitsParterSuccessfulPM, \
           hitsSuccessful_percent, accentedHitsSuccessful_percent, accentedHits_percent, \
           takedownSuccessful_percent, accentedHitsPositionDistanceSuccessful_percent, \
           accentedHitsPositionClinchSuccessful_percent, accentedHitsPositionParterSuccessful_percent, \
           takedowns_to_hits, HitsPositionDistance_to_hits, HitsPositionClinch_to_hits, \
           HitsPositionParter_to_hits


def get_figher_statistics_from_past(df_stats, fighterId, eventDate, fighter1_cols, fighter2_cols):
    '''
    Aggregate statistics from fighter from the past.
    Check both positions 1 and 2.
    '''
    first_pos_df  = df_stats[(df_stats['fighterId_1'] == fighterId) & (df_stats['eventDate.date'] < eventDate)]
    first_pos_df  = pd.DataFrame(first_pos_df[fighter1_cols].values)

    second_pos_df = df_stats[(df_stats['fighterId_2'] == fighterId) & (df_stats['eventDate.date'] < eventDate)]
    second_pos_df = pd.DataFrame(second_pos_df[fighter2_cols].values)

    joined_df = first_pos_df.append(second_pos_df).reset_index(drop=True)

    return joined_df.mean()