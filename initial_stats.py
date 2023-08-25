import ML_stats as ML
from sort_data import Sorter
from data_reader import dataReader
from configs import Config
import numpy as np

if __name__ == '__main__':

    from sklearn import datasets
    cfg = Config('configs.ini')
    cfg.pierre_2 = cfg.get_cfg('Filepaths', 'full_data')
    #data = dataReader(cfg.full_data)
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)
    tmp_data = data.dataset
    tmp_data[tmp_data == '-'] = np.nan
    clean_data = tmp_data.dropna(axis=1)
    #graph = displayData(clean_data)
    #graph.preview_plots(display_dict)
    #graph.visualize_assignments(display_dict)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(data.dataset)

    ##make some heatmaps for an overview
    clean = srt.remove_SD_columns(data.dataset)
    b, p = ML.split_pre_post(data.dataset[clean], 'baseline', '24h post treatment')
    diff = ML.diff_frames(b, p)
    catwalk_data_only = diff.drop(['NumberOfRunsUsedForCalculatingTrialStatistics'], axis=1)
    std_cdo = ML.standard_dataframe(catwalk_data_only)
    cdo_col = catwalk_data_only.columns
    cdo_col = cdo_col[:-1]
    cdo_tukey = ML.yield_pairwise_tukey_across_dataframe(diff, cdo_col)