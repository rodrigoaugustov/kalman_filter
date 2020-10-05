# Importacoes necessarias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm
from itertools import permutations


def find_cointegrated_pairs(dataframe, critial_level = 0.01, modo='simples'):
    keys = permutations(list(dataframe.columns),2) # get the column names
    pairs = [] # initilize the list for cointegration
    passo = 0
    n = dataframe.shape[1] * (dataframe.shape[1]-1)
    for key in keys:

        stock1 = dataframe[key[0]].dropna(axis=0) # obtain the price of "stock1" Y 
        stock2 = dataframe[key[1]].dropna(axis=0) # obtain the price of "stock2" X
        lenght = max(stock1.index[0], stock2.index[0])
        stock1 = stock1[lenght:]
        stock2 = stock2[lenght:]
        
        periodos = [250,210,180,150,120,90,60]
        cointegrados = {} 
        result = sm.tsa.stattools.coint(stock1[-periodos[0]:], stock2[-periodos[0]:])
        
        if result[1] <= critial_level:
            cointegrados[periodos[0]] = result[1]
            if modo == 'complexo':
                for periodo in periodos[1::]:
                    result = sm.tsa.stattools.coint(stock1[-periodo:], stock2[-periodo:])
                    if result[1] <= critial_level*2:
                        cointegrados[periodo] = result[1]
                cointegrados = pd.DataFrame(cointegrados, index={'cointegracoes'})
                contagem = cointegrados.shape[1]
                if contagem >= 3:
                    pairs.append((key[0], key[1], cointegrados[cointegrados.columns[0]][0], contagem, list(cointegrados.columns)))
            else:
                pairs.append((key[0], key[1], cointegrados[periodos[0]]))
                
#        pvalue_matrix[i, j] = min(p_values.values())        
        
        passo = passo + 1
        print(round((passo/n)*100,2))

#    return pvalue_matrix, pairs
    return pairs

### Definindo o filtro de Kalman
def KalmanFilterAverage(x):
  # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)

  # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

# Kalman filter regression
def KalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    em_vars=['observation_covariance'], ######## ajustar 
    transition_covariance=trans_cov)
    kf.em(x, n_iter=5)
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.smooth(y.values)
    return state_means

def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1],0))

    if halflife <= 0:
        halflife = 1
    return halflife

### Definindo a backtest function
def backtest(df, pair, capital=10000):
    #############################################################
    # INPUT:
    # DataFrame of prices
    # s1: the symbol of contract one
    # s2: the symbol of contract two
    # x: the price series of contract one
    # y: the price series of contract two
    # OUTPUT:
    # df1['cum rets']: cumulative returns in pandas data frame
    # sharpe: Sharpe ratio
    # CAGR: Compound Annual Growth Rate
    
    s1 = pair[1]
    s2 = pair[0]
    
    x = df[pair[1]].dropna(axis=0)
    y = df[pair[0]].dropna(axis=0)
    
#    periodo = pair[4]
    periodo = 250
    
    x,y = x[-periodo:], y[-periodo::]     # limitar o window de tempo pra trás
    lenght = max(x.index[0], y.index[0])
    x = x[lenght:].copy()
    y = y[lenght:].copy()
    x.index = pd.to_datetime(x.index)
    y.index = pd.to_datetime(y.index)
    # run regression (including Kalman Filter) to find hedge ratio and then create spread series
    df1 = pd.DataFrame({s2:y[-252::],s1:x[-252::]})
    
    ####################################################################################
#    df1.index = pd.to_datetime(df1.index)
#    
#    hr = []
#    for i in df1.index:
#        state_means = KalmanFilterRegression(KalmanFilterAverage(x[x.index[0]:i]),KalmanFilterAverage(y[y.index[0]:i]))
#        hr.append(- state_means[:,0][-1])
#        
#    df1['hr'] = hr
#  
    ###################################################################################
    state_means = KalmanFilterRegression(KalmanFilterAverage(x),KalmanFilterAverage(y))

    df1['hr'] = - state_means[:,0]
    df1['spread'] = df1[s2] + (df1[s1] * df1['hr'])

    # calculate half life
    halflife = half_life(df1.spread)
    
    window=22

    ### ZCORE UTILIZANDO UM PERIODO ESPECIFICO ###
    meanSpread = df1.spread.rolling(window=window).mean()
    stdSpread = df1.spread.rolling(window=window).std()
    
    
    ### Z-SCORE UTILIZANDO TODO HISTORICO ###
#    meanSpread = df1.spread.mean()
#    stdSpread = df1.spread.std()
               
    df1['zScore'] = (df1.spread-meanSpread)/stdSpread
    
    df1['halflife'] = halflife

    ##############################################################
    # trading logic
    entryZscore = 2
    exitZscore = 1

    
    # CRIANDO O STOP TEMPORAL
#    df1['stoptempo'] = df1.index.shift(halflife, freq='B')
#    for i in range(len(df1.index)-halflife):
#        df1.stoptempo.iloc[i] = df1.index[i+halflife]  
        
        #set up num units long

    df1['long entry'] = ((df1.zScore < - entryZscore) & ( df1.zScore.shift(1) > - entryZscore))
    df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore))
#    try:
#        df1['long exit'].loc[df1.loc[df1['long entry']==True]['stoptempo']] = True
#    except:
#        pass
        
    df1.loc[df1['long entry'],'num units long'] = 1
    df1.loc[df1['long exit'],'num units long'] = 0 
    df1['num units long'].iloc[0] = 0 
    df1['num units long'].ffill(inplace=True)
    df1['num units long'] = df1['num units long'].diff()
    df1['num units long'].loc[df1['num units long'] == 0] = np.nan
    
    #set up num units short 
    df1['short entry'] = ((df1.zScore > entryZscore) & ( df1.zScore.shift(1) < entryZscore))
    df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
#    try:
#        df1['short exit'].loc[df1.loc[df1['short entry']==True]['stoptempo']] = True
#    except:
#        pass
    
    df1.loc[df1['short entry'],'num units short'] = -1
    df1.loc[df1['short exit'],'num units short'] = 0
    df1['num units short'].iloc[0] = 0
    df1['num units short'].ffill(inplace=True)
    df1['num units short'] = df1['num units short'].diff()
    df1['num units short'].loc[df1['num units short'] == 0] = np.nan

    # Preenche os valores do zScore do começo com 0 e retira as colunas que não serão mais uteis
    df1['zScore'].fillna(0, inplace=True)
    df1 = pd.DataFrame(df1[[s2, s1,'hr', 'spread', 'zScore','halflife','num units long', 'num units short']])
    
    # Limpa o dataframe para mostrar somente as linhas onde houve trade
    df_limpo = pd.DataFrame(df1[['num units long', 'num units short']].dropna(axis=0, how='all'))
    df_limpo.fillna(0, inplace=True)
    df_limpo['Entrada/Saida'] = np.nan
    
    # Junta as entradas e saídas Long e Short na mesma coluna
    df_limpo['Trade'] = df_limpo['num units long'] + df_limpo['num units short']
    
    # Cria coluna definindo se foi entrada ou saída
    df_limpo.loc[df_limpo['num units long'] == 1, 'Entrada/Saida'] = 'Entrada'
    df_limpo.loc[df_limpo['num units long'] == -1, 'Entrada/Saida'] = 'Saida'
    df_limpo.loc[df_limpo['num units short'] == -1, 'Entrada/Saida'] = 'Entrada'
    df_limpo.loc[df_limpo['num units short'] == 1, 'Entrada/Saida'] = 'Saida'  

    trades = df_limpo.join(df1[[s2, s1, 'hr','spread', 'zScore', 'halflife']], how='inner')
    
    # Incluindo o tamanho da posição - No momento ajustado somente para Int. Depois testar arrendonando para lote de 100
    trades.loc[trades['Entrada/Saida'] == 'Entrada', s2+'Position'] = round(capital / trades[s2] * trades['Trade'],0) # Filtra as entradas e preenche com a quantidade de entrada baseada no financeiro
    trades[s2+'Position'].ffill(inplace=True) #Preenche a saida com a mesma posicao de entrada
    trades.loc[trades['Entrada/Saida'] == 'Saida', s2+'Position'] = trades[s2+'Position'] * -1 # Inverte o sinal da saida
    
    trades.loc[trades['Entrada/Saida'] == 'Entrada', s1+'Position'] = round(trades[s2+'Position'] * trades['hr'],0) # Filtra as entradas e preenche com a quantidad de entrada baseado no beta
    trades[s1+'Position'].ffill(inplace=True) #Preenche a saida com a mesma posicao de entrada
    trades.loc[trades['Entrada/Saida'] == 'Saida', s1+'Position'] = trades[s1+'Position'] * -1 # Inverte o sinal da saida
    
    # Calculado o financeiro das entradas e saídas
    trades['Financeiro'] = -(trades[s2+'Position'] * trades[s2] + trades[s1+'Position'] * trades[s1])
    
    # Calcula o resultado de cada trade
    trades['ResultadoParcial'] = 0
    trades.loc[trades['Entrada/Saida'] == 'Saida', 'ResultadoParcial'] = trades['Financeiro'].rolling(2).sum()

    # Calcula o resultado acumulado
    trades['Acumulado'] = 0
    trades.loc[trades['Entrada/Saida'] == 'Saida', 'Acumulado'] = trades['Financeiro'].cumsum()
    trades.loc[trades['Acumulado'] == 0, 'Acumulado'] = np.nan
    trades['Acumulado'].ffill(inplace=True)
    trades['Acumulado'].fillna(0, inplace=True)
    
    trades.drop(['num units long','num units short', 'Trade'], axis=1, inplace=True)
    
    trades.columns
    trades = trades.reindex([s2, s1, 'hr', 'spread', 'zScore', 'halflife', 'Entrada/Saida', 
                         s2+'Position',s1+'Position','Financeiro', 'ResultadoParcial', 'Acumulado'],axis=1)
    
    # Criando um dataframe com os trades e toda a série histórica
    df2 = pd.concat([df1[[s2, s1, 'hr', 'spread', 'zScore', 'halflife']], trades], join='outer', axis=1)
    df2.columns = [s2, s1, 'hr', 'spread', 'zScore', 'halflife', 'drop1', 'drop2', 'drop3','drop4','drop5','drop6','Entrada/Saida', 
                         s2+'Position',s1+'Position','Financeiro', 'ResultadoParcial', 'Acumulado']
    drop = df2.columns[6:12]
    df2.drop(drop, inplace=True, axis=1)
    df2['MeanSpread'] = df2.spread.rolling(window=window).mean()
    df2['Acumulado'].fillna(method='backfill', inplace=True)
    df2['Acumulado'].fillna(method='ffill', inplace=True)
    
    return df2, trades

############ Funcao para criar os graficos ###############
def plot_graph(radar, keys, key):
    ativos = keys[key].split('/')
    fig, axes = plt.subplots(figsize=(12,7), nrows=2, ncols=2)
    ax1=plt.subplot(2,2,1)
    plt.plot(radar[keys[key]]['zScore'][-90::])
    spread = radar[keys[key]][['spread','MeanSpread']][-90::]
    spread.plot(ax=axes[0,1])
    radar[keys[key]]['hr'][-90::].plot(ax=axes[1,0])
    radar[keys[key]][-90::].plot(ativos[1], ativos[0], kind='scatter', ax=axes[1,1])
    fig.tight_layout()
    
def gera_pnt(radar):
    from datetime import date

    today = date.today()
    today = today.strftime("%d/%m")
    operacoes = pd.DataFrame()
    for i in radar:
        ativo1 = i.split('/')[0]
        ativo2 = i.split('/')[1]
        qtd1 = int(round(radar[i][ativo1+'Position'].iloc[-1],-2))
        qtd2 = int(round(qtd1 * radar[i]['hr'].iloc[-1],-2))
        if qtd1 > 0:
            sinal1 = 'C'
            sinal2 = 'V'
        else:
            sinal1 = 'V'
            sinal2 = 'C'
        financeiro = int(qtd1 * radar[i][ativo1].iloc[-1] + qtd2 * radar[i][ativo2].iloc[-1])
        #operacao = [[ativo1,ativo2,sinal1,sinal2,abs(qtd1),abs(qtd2),3,3,-financeiro]] ## PNT
        ret_esperado = retorno_esperado(radar[i])              
        operacao = [[today, ativo1,ativo2,sinal1,sinal2,abs(qtd1),abs(qtd2), 
                     radar[i]['halflife'].iloc[-1], -financeiro, ret_esperado, radar[i]['zScore'].iloc[-1], radar[i][ativo1].iloc[-1], radar[i][ativo2].iloc[-1]]] ## Excel
        operacoes = operacoes.append(operacao)
        
    return operacoes

def atualiza(coints_kf, hoje, capital=10000):
    import numpy as np
    for i in coints_kf:
        ativo1 = i.split('/')[0]
        ativo2 = i.split('/')[1]
        coints_kf[i][ativo1].iloc[-1] = hoje[ativo1][0]
        coints_kf[i][ativo2].iloc[-1] = hoje[ativo2][0]
        coints_kf[i]['spread'].iloc[-1] = hoje[ativo1][0] + (hoje[ativo2][0] * coints_kf[i]['hr'].iloc[-1])
        
        window=22
        
        ###  Janela limitada ###
        meanSpread = coints_kf[i].spread.rolling(window=window).mean()
        stdSpread = coints_kf[i].spread.rolling(window=window).std()
        coints_kf[i]['zScore'].iloc[-1] = (coints_kf[i].spread.iloc[-1] - meanSpread.iloc[-1])/stdSpread.iloc[-1]

        ### Janela Fixa ### 
#        meanSpread = coints_kf[i].spread.mean()
#        stdSpread = coints_kf[i].spread.std()          
#        coints_kf[i]['zScore'] = (coints_kf[i].spread - meanSpread)/stdSpread

        if coints_kf[i]['zScore'].iloc[-1] >= 2 or coints_kf[i]['zScore'].iloc[-1] <= -2:
            coints_kf[i]['Entrada/Saida'].iloc[-1] = 'Entrada'
            coints_kf[i].iloc[-1, coints_kf[i].columns.get_loc(ativo1+'Position')] = round(capital / hoje[ativo1][0] * - np.sign(coints_kf[i]['zScore'].iloc[-1]),-2)
            coints_kf[i][ativo2+'Position'].iloc[-1] = round(coints_kf[i][ativo1+'Position'].iloc[-1] * coints_kf[i]['hr'].iloc[-1],-2)
            coints_kf[i].Financeiro.iloc[-1] = -(coints_kf[i][ativo1+'Position'].iloc[-1] * coints_kf[i][ativo1].iloc[-1] + coints_kf[i][ativo2+'Position'].iloc[-1] * coints_kf[i][ativo2].iloc[-1])
        else:
            coints_kf[i].replace(coints_kf[i]['Entrada/Saida'].iloc[-1],np.nan, inplace=True)
    radar = {}
    for key in coints_kf:
        if coints_kf[key]['Entrada/Saida'][-1] == 'Entrada':
            radar[key] = coints_kf[key]
    for i in radar:
        radar[i]['BetaRotation'] = abs(radar[i]['hr'].rolling(window=window).mean())
    
    entrada = gera_pnt(radar)
    
    return coints_kf, radar, entrada

def retorno_esperado(df1):
    df = df1.copy()
    
    def zscore_function(y_value, df1):
        
        y = df.columns[0]
        x = df.columns[1]
            
        df.replace(df.iloc[-1][y], y_value, inplace=True)
        
        window=22
            
        df['spread'] = df[y] + (df[x] * df['hr'])    
        
        meanSpread = df.spread.rolling(window=window).mean()
        stdSpread = df.spread.rolling(window=window).std()
        
#        meanSpread = df.spread.mean()
#        stdSpread = df.spread.std()
        
        df['zScore'] = (df.spread-meanSpread) / stdSpread
        zscore_obtido = df['zScore'].iloc[-1]
        
        return abs(zscore_obtido)
    
    ativo_y = df.columns[0]
    ativo_x = df.columns[1]
    
    y_value = df.iloc[-1][0]
    goal = abs(df.zScore.iloc[-1]) - 1
    res = zscore_function(y_value, df)
    
    while res > goal:
        if df[ativo_y+'Position'].iloc[-1] < 0:
            y_value = y_value * 0.995
        else:
            y_value = y_value* 1.005
        res = zscore_function(y_value, df)
        
    

    
    y = y_value
    x = df.iloc[-1][1]
    
    df[ativo_y+'Saida'] = y
    df[ativo_x+'Saida'] = x
    
    df['Financeiro Saida'] = df[ativo_y+'Position'] * df[ativo_y+'Saida'] + df[ativo_x+'Position'] * df[ativo_x+'Saida']
    df['Esperado'] = df['Financeiro Saida'] + df.Financeiro
    
    valor_esperado = df.iloc[-1]['Esperado']
    
    return valor_esperado


