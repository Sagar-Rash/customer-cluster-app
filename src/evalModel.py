def evaluateModel(train_pred, y_train, test_pred, y_test, evalType, print_vals = False):
    
    # Evaluate model based on different criteria
    
    if evalType.lower() == "mae":
        from sklearn.metrics import mean_absolute_error
        try:
            # MAE for train model
            train_mae = mean_absolute_error(y_train, train_pred)
            # MAE for test model
            test_mae = mean_absolute_error(y_test, test_pred)
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        if print_vals:
            print('Train error is', train_mae)
            print('Test error is', test_mae)
    
    elif evalType.lower() == "acc":
        # Accuracy score
        from sklearn.metrics import accuracy_score
        
        try:
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        
        if print_vals:
            print("Train accuracy is", train_acc)      
            print("Test accuracy is", test_acc)
    
    elif evalType.lower() == "conf":
        from sklearn.metrics import confusion_matrix
        # Confusion Matrix
        try:
            train_conf = confusion_matrix(y_train, train_pred)
            test_conf = confusion_matrix(y_test, test_pred)
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        
        if print_vals:
            print("Train confusion matrix is", train_conf)
            print("Test confusion matrix is", test_conf)
        
    
    elif evalType.lower() == "prec":
        from sklearn.metrics import precision_score
        # Precision score
        try:
            train_prec = precision_score(y_train, train_pred)
            test_prec = precision_score(y_test, test_pred)
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        
        if print_vals:
            print("Train precision score is", train_prec)
            print("Test precision score is", test_prec)
            
    elif evalType.lower() in ["rsq","r2"]:
        from sklearn.metrics import r2_score
        # r squared
        try:
            train_r2 = r2_score(y_train,train_pred)
            test_r2 = r2_score(y_test,test_pred)
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        
        print('Train R-squared score:', train_r2)
        print('Test R-squared score:', test_r2)
        
    elif evalType.lower() == "rmse":
        # Root Mean Squared Error
        from numpy import sqrt
        from sklearn.metrics import mean_squared_error
        try:
            train_rmse = sqrt(mean_squared_error(y_train,train_pred))
            test_rmse = sqrt(mean_squared_error(y_test,test_pred))
        except Exception as e:
            print(f"Error occurred: {e}")
            return -1
        
        print('Train Root Mean Squared Error (RMSE):', train_rmse)
        print('Test Root Mean Squared Error (RMSE):', test_rmse)