def backwardPass(X, Y, P, XList, WList, lamb, batch_start=0, batch_size=20, doBatchNormalization=False, S_hatList=None, meanList=None, varList=None, S_List=None, gammaList=None, betaList=None):
    if doBatchNormalization:
        X_batch, Y_batch, P_batch = X, Y, P # ?
        # Walk backwards through the intermediary calculations
        XList.reverse(), WList.reverse(),gammaList.reverse(),betaList.reverse(),varList.reverse(),meanList.reverse(), S_hatList.reverse(),S_List.reverse()
        W_last = WList.pop() # Get the last weight matrix for the initial computation
        G_vec = - (Y_batch-P_batch) # Initialize G_vec
        dJdW_list, dJdb_list, dJdgamma_list, dJdbeta_list = [], [], [], []
        a = 0
        for index, [Xk, W] in enumerate(zip(XList, WList)): # Loop through all intermediary steps
            if index == 0: # First computation, last gradient
                Xk_batch = Xk
                dJdW = ((G_vec @ Xk_batch.T))/batch_size
                dJdb = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size
                dJdW_list.append(dJdW + 2*lamb*W), dJdb_list.append(dJdb)
                a += 1
                # print(str(a) + ': \n Data: ', Xk.shape,'\n W: ', W.shape)

                G_vec = W.T @ G_vec # Update G_vec for the next computation
                Xk_batch[Xk_batch<0] = 0
                Xk_batch[Xk_batch>0] = 1
                G_vec = np.multiply(G_vec, Xk_batch)
            else: # Compute the gradient with respect to the batch normalization parameters
                Xk_batch = Xk
                dJdgamma = np.sum(np.multiply(G_vec, S_hatList[index-1]), axis=1)[:, np.newaxis]/batch_size # 25a
                dJdbeta = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size # 25b
                G_vec = G_vec * gammaList[index-1] # 26
                G_vec = batchNormBackPass(G_vec, S_List[index], meanList[index-1], varList[index-1]) # 27
                dJdW = ((G_vec @ Xk.T))/batch_size + 2*lamb*W
                dJdb = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size
                dJdW_list.append(dJdW), dJdb_list.append(dJdb), dJdgamma_list.append(dJdgamma), dJdbeta_list.append(dJdbeta)
                a += 1
                
                G_vec = W.T @ G_vec # Update G_vec for the next computation
                Xk_batch[Xk_batch<0] = 0
                Xk_batch[Xk_batch>0] = 1
                G_vec = np.multiply(G_vec, Xk_batch)

        # Get the final gradient from the input X, (first gradient)
        dJdgamma = np.sum(np.multiply(G_vec, S_hatList[-1]), axis=1)[:, np.newaxis]/batch_size # 25a
        dJdbeta = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size # 25b
        G_vec = G_vec * gammaList[-1] # 26
        G_vec = batchNormBackPass(G_vec, S_List[-1], meanList[-1], varList[-1]) # 27
        dJdW = ((G_vec @ X_batch.T))/batch_size + 2*lamb*W_last
        dJdb = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size
        dJdW_list.append(dJdW), dJdb_list.append(dJdb), dJdgamma_list.append(dJdgamma), dJdbeta_list.append(dJdbeta)
        a += 1
        
        dJdW_list.reverse(), dJdb_list.reverse(), dJdgamma_list.reverse(), dJdbeta_list.reverse()
        return dJdW_list, dJdb_list, dJdgamma_list, dJdbeta_list
    else:
        X_batch, Y_batch, P_batch = X, Y, P, 
        
        XList.reverse() # Walk backwards through the intermediary calculations
        WList.reverse()
        W_last = WList.pop() # Get the last weight matrix for the initial computation
        G_vec = - (Y_batch-P_batch) # Initialize G_vec
        dJdW_list = []
        dJdb_list = []
        for Xk, W in zip(XList, WList): # Loop through all intermediary steps
            Xk_batch = Xk
            dJdW = ((G_vec @ Xk_batch.T))/batch_size
            dJdb = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size
            dJdW_list.append(dJdW + 2*lamb*W)
            dJdb_list.append(dJdb)
            
            G_vec = W.T @ G_vec # Update G_vec for the next computation
            Xk_batch[Xk_batch<0] = 0
            Xk_batch[Xk_batch>0] = 1
            G_vec = np.multiply(G_vec, Xk_batch)
        
        # Get the final gradient from the input X
        dJdW = ((G_vec @ X_batch.T))/batch_size
        dJdb = np.sum(G_vec, axis=1)[:, np.newaxis]/batch_size
        dJdW_list.append(dJdW + 2*lamb*W_last)
        dJdb_list.append(dJdb)
        dJdW_list.reverse() 
        dJdb_list.reverse()
        
        return dJdW_list, dJdb_list, None, None
