def strength_seasonal_trend(ts, season=12):
    '''
    La funzione calcola, sfruttando il principio di decomposizione, la forza della
    stagionalità e del trend
    
    Parameters
    ----------
    ts : pandas.Series
        Serie temporale

    Returns
    -------
    strength_seasonal : float
        Grado della stagionalità
    strength_seasonal_trend : float
        Grado del trend

    '''
    # Decomponiamo la serie temporale
    decomposition = seasonal_decompose(ts, period=season)
    
    # Estraiamo le componenti elementari
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Calcoliamo la forza mettendo in rapporto la varianza dei residui con la
    # Serie de-stagionalizzata (per la forza del trend) o la serie de-trendizzata (per la forza stagionale)
    
    strength_trend = max(0, 1 - residual.var()/(trend + residual).var())
    strength_seasonal = max(0, 1 - residual.var()/(seasonal + residual).var())
    
    return (strength_seasonal, strength_trend)

def print_seasons(dataset):
    _max = float("-inf")
    _best_season = 0
    
    # 19 14 8
    
    index = 1
    for column in dataset.transpose():
        if index in {8, 14, 19}:
            for i in range(1, 500):
                print("index =,", index, "i=", i)
                strength = strength_seasonal_trend(pd.Series(column))[0]
                if _max < strength:
                    _best_season = i
                    _max = strength
            
            print("best:", _best_season, "colonna index:", index)
        
        index += 1