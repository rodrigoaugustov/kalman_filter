# Filtro de Kalman para Pair Trading

Algoritmo para aplicação do Filtro de Kalman para operações de Pair Trading (Long&Short)

Basicamente um mini trade-system para localizar operações que estão dando entrada.

Primeiro filtramos os pares que são cointegrados entre si, e para esses pares, é aplicado o filtro de kalman.

Com base no z-Score é passado um radar em busca dos pares que estão dando entrada.

Ao final é gerado uma variável 'entrada', que pode ser utilizada para utilização no Robo PnT.

Código ainda em fase de desenvolvimento, muita coisa precisa ser melhorada/otimizada.

Para base de dados verificar: https://github.com/rodrigoaugustov/cotacoes_tryd 

De preferência limitar o período máximo de dados para diminuir o tempo de execução.
