
def corrplots(input_csv, input_params, method="spearman", print_details = False):
    # display correlation plot, descriptive statistics, and pairplot
    # return dataframe of data
    from pandas import read_csv
    from seaborn import pairplot
    from matplotlib.pyplot import show
    if method.lower() not in ["pearson", "spearman"]:
        print("method should be Pearson or Spearman.")
        return 0
    else:
        # input_params should have at least 4 values
        try:
            corr_df = read_csv(input_csv)
            if print_details:
                print("The Descriptive statistics for this dataset can be seen below:")
                print(corr_df.describe())
                print('\n')
                print("The Correlation matrix for this dataset can be seen below:")
                print(corr_df.corr(numeric_only=True, method=method))
                print('\n')
                print('The following is a pairplot:')
                pairplot(corr_df[input_params[0:3]])
                show()
                print('\n')
                print('The following is a second pairplot:')
                pairplot(corr_df[input_params[1:4]])
                show()
            return corr_df
        except Exception as e:
            print(f"Error occurred: {e}")
            return 0
        