import pandas
import numpy as np

fdy = [0, 365, 731, 1096] # the first date of years from 2019 to 2022


def get_avg_ratio(df):

    peak_capacity = df["尖峰供電能力(MW)"].values
    peak_load = df["尖峰負載(MW)"].values
    # peak energy capacity in Jan and Feb in past three years
    c19 = np.sum(peak_capacity[fdy[0]:fdy[0]+59])
    c20 = np.sum(peak_capacity[fdy[1]:fdy[1]+59])
    c21 = np.sum(peak_capacity[fdy[2]:fdy[2]+59])
    c22 = np.sum(peak_capacity[fdy[3]:])
    # average peak energy capacity
    c_average = (c19 + c20 + c21) / 177

    # peak energy load in Jan and Feb in past three years
    l19 = np.sum(peak_load[fdy[0]:fdy[0]+59])
    l20 = np.sum(peak_load[fdy[1]:fdy[1]+59])
    l21 = np.sum(peak_load[fdy[2]:fdy[2]+59])
    l22 = np.sum(peak_load[fdy[3]:])
    # average peak energy load
    l_average = (l19 + l20 + l21) / 177

    # We calculate the average of the first two month this year and 
    # compare it to the average of the first two month in past three
    # year. Using the ratio to calculate if this year use more energy 
    # or less.
    c_average22 = c22 / 59
    l_average22 = l22 / 59

    return [c_average, l_average, c_average22, l_average22]

def get_week_ratio(df):

    peak_capacity = df["尖峰供電能力(MW)"].values
    peak_load = df["尖峰負載(MW)"].values

    week_count = [0 for i in range(7)]
    week_cap = [0 for i in range(7)]
    week_load = [0 for i in range(7)]
    days = 1 # 2019/1/1 Tuesday
    for i in range(len(peak_load)):
        week_count[days] += 1
        week_cap[days] += peak_capacity[i]
        week_load[days] += peak_load[i]
        days = (days+1) % 7
        
    # Each day in week has different amount of energy usage, we calculate
    # the average in past three year, and use it to transform usage
    # between different days in a week. 
    week_cap = np.array(week_cap)/np.array(week_count)
    week_load = np.array(week_load)/np.array(week_count)

    return [week_cap, week_load]

def make_predict(ca, la, ca22, la22, wc, wl, df):
    week_on_day = [2, 3, 4, 5, 6, 6, 6, 2, 3, 4, 5, 6, 0, 1]
    offset = 88 # date offset = 31 + 29 + 29 - 1
    peak_capacity = df["尖峰供電能力(MW)"].values
    peak_load = df["尖峰負載(MW)"].values
    pred = []

    for i in range(14):
        w = week_on_day[i]
        cap_ref = [peak_capacity[fdy[0] + offset], peak_capacity[fdy[1] + offset + 1],
                    peak_capacity[fdy[2] + offset]]
        load_ref = [peak_load[fdy[0]+offset], peak_load[fdy[1]+offset+1], peak_load[fdy[2]+offset]]
        week_ref = [(offset + 1 + fdy[0]) % 7, (offset + 2 + fdy[1]) % 7, (offset + 1 + fdy[2]) % 7]
        cap_base = (cap_ref[0] * wc[w] / wc[week_ref[0]] + cap_ref[1] * wc[w] / wc[week_ref[1]] +
                    cap_ref[2] * wc[w] / wc[week_ref[2]]) / 3
        load_base = (load_ref[0] * wl[w] / wl[week_ref[0]] + load_ref[1] * wl[w] / wl[week_ref[1]] +
                    load_ref[2] * wl[w] / wl[week_ref[2]]) / 3
        cap_pred = cap_base / ca * ca22
        load_pred = load_base / la * la22
        offset += 1
        pred.append(int(cap_pred - load_pred))

    return pred

if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    df = pandas.read_csv(args.training)
    date = ['20220330','20220331','20220401','20220402','20220403','20220404','20220405',
            '20220406','20220407','20220408','20220409','20220410','20220411','20220412']
    cap_avg, load_avg, cap_avg_22, load_avg_22 = get_avg_ratio(df)
    week_cap_ratio, week_load_ratio = get_week_ratio(df)
    pred = make_predict(cap_avg, load_avg, cap_avg_22, load_avg_22, week_cap_ratio, week_load_ratio, df)

    result = pandas.DataFrame()
    result['date'] = date
    result['operating_reserve(MW)'] = pred
    result.to_csv(args.output, index=0)
    