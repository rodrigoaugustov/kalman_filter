from kalman_functions import find_cointegrated_pairs, backtest, plot_graph, gera_pnt, atualiza
import pandas as pd

# Importa as cotacoes
data = pd.read_csv('E:\OneDrive\Documentos\Quant\Cotacoes\cotacoes.csv', index_col=0) ## Se precisar alterar a pasta
data.drop(data.index[0], inplace=True)
data = data[-252::]
a = list((data!=0).all())
data = data.T
data = data[a].T
col = list(data.columns)

dataframe = data.copy()

# Roda a função para verificar os pares cointegrados
pairs = find_cointegrated_pairs(dataframe, critial_level = 0.01)

# Implementa o filtro de Kalman para os pares cointegrados e roda o backtest
coints_kf = {}
trades = {}
for pair in pairs:
    coints_kf[pair[0]+'/'+pair[1]], trades[pair[0]+'/'+pair[1]] = backtest(dataframe,pair)
    print(pair) ## Como são muitos ativos, deixo o print ativado apenas para acompanhar o andamento
    
# Filtra operações com entrada ativa
radar = {}
for key in coints_kf:
    if coints_kf[key]['Entrada/Saida'][-1] == 'Entrada':
        radar[key] = coints_kf[key]
        
entrada = gera_pnt(radar)