from sample import load_dataset


_, df_target = load_dataset()
df_tmp = df_target.reset_index()
idx = df_tmp[df_tmp['obsid'] == 284201150].index
print(idx)
