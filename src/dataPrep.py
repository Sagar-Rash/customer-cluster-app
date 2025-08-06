

def createTrainTest(csvPath, target, stratifyBy = None, test_sz = 0.2, print_vals = False):
    # Create train-test data, given dataset, target, and other values.    
    from pandas import read_csv, set_option
    from sklearn.model_selection import train_test_split

    set_option('display.max_columns', 50)
    
    try:
        df = read_csv(csvPath)
        # create x, y, and features list
        x = df.drop(target, axis=1)
        features = list(x.columns)
        y = df[target]
        # If stratifyBy is not the target, then set the attribute to be x[stratifyBy] to match syntax
        if stratifyBy == None:
            stratifyAttr = None
        elif stratifyBy != target:
            stratifyAttr = x[stratifyBy]
        else:
            stratifyAttr = y
        # Split
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_sz, stratify=stratifyAttr)
        
        if print_vals:
            # If instructed to print values, then print the values. Otherwise, do not (e.g. in Streamlit)
            print("Head:")
            print(df.head(5))
            print('\n')
            print("Tail:")
            print(df.tail(5))
            print('\n')
            print("Dataframe Shape:")
            print(df.shape)
            print('\n')
            print("Shape of Split Data:")
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, x_test, y_train, y_test, features
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0
    
def scaleData(xtrain,xtest):
    from sklearn.preprocessing import MinMaxScaler
    # Scale data with minmaxscaler
    scale = MinMaxScaler()
    
    try:    
        xtrain_scaled = scale.fit_transform(xtrain)
        xtest_scaled = scale.transform(xtest)
        
        return xtrain_scaled, xtest_scaled
    except Exception as e:
            print(f"Error occurred: {e}")
            return 0