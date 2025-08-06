

def pickleModel(strModelName, strMode, objModel = None, pickleAction='save'):
    
    from pickle import dump, load
    # Save model to strModelName (filepath)
    # objModel = model name in notebook/code
    # strMode = 'wb', 'rb', etc.
    try:
        if pickleAction.lower() == 'save' and objModel != None:
            dump(objModel, open(strModelName, strMode))
        
        elif pickleAction.lower() == 'load':
            # Load the pickled model
            loadedModel = load(open(strModelName, strMode))
            
            return loadedModel
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0
    

def predict(loadedModel, lstInputs):
    # predict values with loaded model and input list (input should be list)
    try:
        prediction = loadedModel.predict([lstInputs])
        print(prediction)
        return prediction
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0